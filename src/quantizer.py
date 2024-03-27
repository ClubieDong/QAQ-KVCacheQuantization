import math
import torch
import numpy as np
from models import CausalLM
from scipy.stats import norm
from itertools import product
from functools import cached_property
from typing import Literal, Optional, Any


AttentionType = list[torch.Tensor]
QuantizationLevels = Literal["no-quantization", "token", "layer", "head"]
QuantizationMethods = Literal["uniform", "normal"]


class Quantizer:
    def __init__(self,
                 # Key cache or value cache
                 key_or_value_cache: Optional[Literal["key", "value"]] = None,
                 # no-quantization, token-level, layer-level, or head-level
                 level: Optional[QuantizationLevels] = None,
                 # True: assume cache is already zero-centered, only scale it
                 # False: zero-center cache and then scale it
                 symmetric: Optional[bool] = None,
                 # uniform: assume normalized cache values obbey uniform distribution between max value and min value
                 # normal: assume normalized cache values obbey standard normal distribution
                 method: Optional[QuantizationMethods] = None,
                 # Percentage of outliers (including both lowest and highest)
                 outliers_ratio: Optional[float] = None,
                 # Whether to enable attention-aware quantization
                 use_attentions: Optional[bool] = None,
                 # (only applicable for uniform quantization)
                 # The uniform quantization bits
                 n_bits_uniform: Optional[int] = None,
                 # (only applicable for attention-aware quantization)
                 # Use the last n rows of attentions to calculate quantization bits
                 last_n_attentions: Optional[int] = None,
                 # (only applicable for attention-aware quantization)
                 # Target quantization error
                 target_quantization_error: Optional[float] = None,
                 # (only applicable for attention-aware quantization)
                 # Minimum allowed quantization bits
                 n_bits_min: Optional[int] = None,
                 # (only applicable for attention-aware quantization)
                 # Maximum allowed quantization bits
                 n_bits_max: Optional[int] = None,
                 # (only applicable for attention-aware quantization of key cache)
                 # 2-norm of query tensor used in the formula
                 q_norm: Optional[float] = None):
        # Set key_or_value_cache
        assert key_or_value_cache is not None
        self.key_or_value_cache = key_or_value_cache
        # Early exit for no quantization
        assert level is not None
        self.level = level
        if level == "no-quantization":
            return
        # Set level
        if level == "token":
            self.quantize_dims = (-3, -2, -1)
        elif level == "layer":
            self.quantize_dims = (-2, -1)
        elif level == "head":
            self.quantize_dims = (-1,)
        # Set symmetric
        assert symmetric is not None
        self.symmetric = symmetric
        # Set outliers_ratio
        assert outliers_ratio is not None
        self.outliers_ratio = outliers_ratio
        # Set use_attentions:
        assert use_attentions is not None
        self.use_attentions = use_attentions
        if use_attentions:
            # Set last_n_attentions
            assert last_n_attentions is not None
            assert last_n_attentions > 0
            self.last_n_attentions = last_n_attentions
            # Set target_quantization_error
            assert target_quantization_error is not None
            assert target_quantization_error > 0.0
            self.target_quantization_error = target_quantization_error
            # Set n_bits_min
            assert n_bits_min is not None
            assert 0 <= n_bits_min <= 16
            self.n_bits_min = n_bits_min
            # Set n_bits_max
            assert n_bits_max is not None
            assert n_bits_min <= n_bits_max <= 16
            self.n_bits_max = n_bits_max
            if self.key_or_value_cache == "key":
                # Set q_norm
                assert q_norm is not None
                assert q_norm > 0
                self.q_norm = q_norm
        else:
            # Set n_bits_uniform
            assert n_bits_uniform is not None
            assert 0 <= n_bits_uniform <= 16
            self.n_bits_uniform = n_bits_uniform
        # Set method
        assert method is not None
        self.method_name = method
        if method == "uniform":
            self.quantization_method = self._uniform_quantize
        elif method == "normal":
            self.quantization_method = self._normal_quantize

    def set_dtype_and_device(self, dtype: torch.dtype, device: torch.device):
        self.dtype = dtype
        self.device = device
        if self.level != "no-quantization" and self.method_name == "normal":
            if self.use_attentions:
                n_bits_range = range(self.n_bits_min, self.n_bits_max+1)
            else:
                n_bits_range = range(self.n_bits_uniform, self.n_bits_uniform+1)
            self.normal_quantiles_upper_bound = {
                n: torch.tensor(norm.ppf(np.arange(0, 1, 1/(2**n)) + 1/(2**n)), dtype=dtype, device=device)
                for n in n_bits_range
            }
            self.normal_quantiles_center = {
                n: torch.tensor(norm.ppf(np.arange(0, 1, 1/(2**n)) + 0.5/(2**n)), dtype=dtype, device=device)
                for n in n_bits_range
            }

    @cached_property
    def params(self) -> dict[str, Any]:
        res: dict[str, Any] = {}
        res["key_or_value_cache"] = self.key_or_value_cache
        res["level"] = self.level
        if self.level == "no-quantization":
            return res
        res["symmetric"] = self.symmetric
        res["method_name"] = self.method_name
        res["outliers_ratio"] = self.outliers_ratio
        res["use_attentions"] = self.use_attentions
        if self.use_attentions:
            res["n_bits_min"] = self.n_bits_min
            res["n_bits_max"] = self.n_bits_max
            res["last_n_attentions"] = self.last_n_attentions
            res["target_quantization_error"] = self.target_quantization_error
            if self.key_or_value_cache == "key":
                res["q_norm"] = self.q_norm
        else:
            res["n_bits_uniform"] = self.n_bits_uniform
        return res

    def _calc_quantization_bits(self, attentions: AttentionType, cache: torch.Tensor, outlier_mask: torch.Tensor) -> torch.Tensor:
        # cache/outlier_mask.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        n_batch, seq_len, n_layer, n_head, _ = cache.shape
        if not self.use_attentions:
            if self.level == "token":
                shape = (n_batch, seq_len)
            elif self.level == "layer":
                shape = (n_batch, seq_len, n_layer)
            elif self.level == "head":
                shape = (n_batch, seq_len, n_layer, n_head)
            return torch.ones(shape, dtype=torch.int64, device=self.device) * self.n_bits_uniform
        if self.key_or_value_cache == "key":
            max_error = math.sqrt(12.0 / self.q_norm * math.log(seq_len**3/(seq_len-1) * self.target_quantization_error**2 + 1))
            # max_error.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        elif self.key_or_value_cache == "value":
            attentions = torch.stack(attentions)
            # attentions.shape: (n_layer, n_batch, n_head, seq_len, seq_len)
            attentions = attentions[:, :, :, -self.last_n_attentions:, :]
            # attentions.shape: (n_layer, n_batch, n_head, last_n_attentions, seq_len)
            attentions = attentions.permute(1, 4, 0, 2, 3)
            # attentions.shape: (n_batch, seq_len, n_layer, n_head, last_n_attentions)
            attentions = attentions.amax(dim=self.quantize_dims)
            # attentions.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
            max_error = math.sqrt(12.0 / seq_len) * self.target_quantization_error / attentions
            # max_error.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        cache = torch.masked.masked_tensor(cache, torch.logical_not(outlier_mask))
        # NOTE: PyTorch's bug: https://github.com/pytorch/pytorch/issues/115624
        quantize_dims = [x + len(cache.shape) for x in self.quantize_dims]
        if self.method_name == "uniform":
            if self.symmetric:
                max_value = cache.abs().amax(dim=quantize_dims)
                scale_value = 2 * max_value
            else:
                max_value = cache.amax(dim=quantize_dims)
                min_value = cache.amin(dim=quantize_dims)
                scale_value = max_value - min_value
            assert scale_value.get_mask().all()
            scale_value = scale_value.get_data()
            # scale_value.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
            n_bits = torch.log2(scale_value / (2 * max_error) + 1)
            # n_bits.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        elif self.method_name == "normal":
            raise NotImplementedError()
        n_bits = torch.clamp(torch.ceil(n_bits).to(torch.int64), self.n_bits_min, self.n_bits_max)
        # n_bits.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        # The last (last_n_attentions-1) tokens do not have enough history attentions so we do not quantize them
        if self.last_n_attentions > 1:
            n_bits[:, -self.last_n_attentions+1:] = self.n_bits_max
        return n_bits

    def _calc_outlier_mask(self, cache: torch.Tensor) -> torch.Tensor:
        if self.outliers_ratio == 0.0:
            return torch.zeros_like(cache, dtype=torch.bool, device=self.device)
        # cache.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        cache_flat = cache.flatten(start_dim=-len(self.quantize_dims))
        # cache_flat.shape: (n_batch, seq_len, n_layer*n_head*embed_size_per_head) or (n_batch, seq_len, n_layer, n_head*embed_size_per_head) or (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        lower_threshold = torch.kthvalue(cache_flat, k=int(math.ceil(self.outliers_ratio/2 * cache_flat.shape[-1])), dim=-1).values
        upper_threshold = torch.kthvalue(cache_flat, k=int(math.floor((1.0 - self.outliers_ratio/2) * cache_flat.shape[-1])), dim=-1).values
        # lower_threshold/upper_threshold.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        lower_threshold = lower_threshold.view(*lower_threshold.shape, *((1,)*len(self.quantize_dims)))
        upper_threshold = upper_threshold.view(*upper_threshold.shape, *((1,)*len(self.quantize_dims)))
        # lower_threshold/upper_threshold.shape: (n_batch, seq_len, 1, 1, 1) or (n_batch, seq_len, n_layer, 1, 1) or (n_batch, seq_len, n_layer, n_head, 1)
        mask = (cache <= lower_threshold) | (cache > upper_threshold)
        # mask.shape = (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        return mask

    # Returns (normalized cache, mean value, scale value)
    def _normalize(self, cache: torch.Tensor, method: Literal["minmax", "std"], n_bits: int, outlier_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # cache/outlier_mask.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        cache = torch.masked.masked_tensor(cache, torch.logical_not(outlier_mask))
        if self.symmetric:
            mean_value = torch.zeros((1,)*cache.dim(), dtype=self.dtype, device=self.device)
            if method == "minmax":
                max_value = cache.abs().amax(dim=self.quantize_dims, keepdim=True)
                scale_value = 2 * max_value / (2 ** n_bits)
            elif method == "std":
                scale_value = cache.std(dim=self.quantize_dims, keepdim=True)
        else:
            mean_value = cache.mean(dim=self.quantize_dims, keepdim=True)
            if method == "minmax":
                max_value = cache.amax(dim=self.quantize_dims, keepdim=True)
                min_value = cache.amin(dim=self.quantize_dims, keepdim=True)
                scale_value = (max_value - min_value) / (2 ** n_bits)
            elif method == "std":
                scale_value = cache.to(torch.float64).std(dim=self.quantize_dims, keepdim=True).to(self.dtype)
        assert mean_value.get_mask().all()
        assert scale_value.get_mask().all()
        cache = cache.get_data()
        mean_value = mean_value.get_data()
        scale_value = scale_value.get_data()
        # mean_value/scale_value.shape: (n_count, n_layer/1, n_head/1, embed_size_per_head/1) or (n_count, n_head/1, embed_size_per_head/1), (n_count, embed_size_per_head/1)
        normalized_cache = (cache - mean_value) / scale_value
        # normalized_cache.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        return normalized_cache, mean_value, scale_value

    def _denormalize(self, normalized_cache: torch.Tensor, mean_value: torch.Tensor, scale_value: torch.Tensor) -> torch.Tensor:
        return normalized_cache * scale_value + mean_value

    def _uniform_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        # cache/outlier_mask.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        normalized_cache, mean_value, scale_value = self._normalize(cache, "minmax", n_bits, outlier_mask)
        quantized_cache = torch.clamp(torch.round(normalized_cache).to(torch.int32), -(2 ** (n_bits-1)), 2 ** (n_bits-1) - 1)
        dequantized_cache = quantized_cache.to(self.dtype)
        denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)
        # denormalized_cache.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        return torch.where(outlier_mask, cache, denormalized_cache)

    def _normal_quantize(self, cache: torch.Tensor, n_bits: int, outlier_mask: torch.Tensor) -> torch.Tensor:
        # cache/outlier_mask.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        normalized_cache, mean_value, scale_value = self._normalize(cache, "std", n_bits, outlier_mask)
        quantized_cache = torch.searchsorted(self.normal_quantiles_upper_bound[n_bits], normalized_cache.contiguous())
        dequantized_cache = self.normal_quantiles_center[n_bits][quantized_cache]
        denormalized_cache = self._denormalize(dequantized_cache, mean_value, scale_value)
        # denormalized_cache.shape: (n_count, n_layer, n_head, embed_size_per_head) or (n_count, n_head, embed_size_per_head), (n_count, embed_size_per_head)
        return torch.where(outlier_mask, cache, denormalized_cache)

    # Returns (quantized kvcache, average n_bits)
    def quantize(self, cache: torch.Tensor, attentions: AttentionType) -> tuple[torch.Tensor, float]:
        if self.level == "no-quantization":
            return cache, torch.finfo(self.dtype).bits
        # cache.shape: (n_layer, n_batch, n_head, seq_len, embed_size_per_head)
        cache = cache.permute(1, 3, 0, 2, 4)
        # cache.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        outlier_mask = self._calc_outlier_mask(cache)
        # outlier_mask.shape: (n_batch, seq_len, n_layer, n_head, embed_size_per_head)
        n_bits = self._calc_quantization_bits(attentions, cache, outlier_mask)
        # n_bits.shape: (n_batch, seq_len) or (n_batch, seq_len, n_layer) or (n_batch, seq_len, n_layer, n_head)
        average_n_bits = n_bits.mean(dtype=self.dtype).item()
        average_n_bits = average_n_bits * (1 - self.outliers_ratio) + torch.finfo(self.dtype).bits * self.outliers_ratio
        n_bits_min, n_bits_max = n_bits.min().item(), n_bits.max().item()
        cache = cache.clone()
        for n in range(n_bits_min, n_bits_max+1):
            indices = torch.where(n_bits == n)
            cache[indices] = self.quantization_method(cache[indices], n_bits=n, outlier_mask=outlier_mask[indices])
        cache = cache.permute(2, 0, 3, 1, 4)
        # cache.shape: (n_layer, n_batch, n_head, seq_len, embed_size_per_head)
        return cache, average_n_bits

    def calc_quantized_cache_size_per_token(self, average_n_bits: float, model: CausalLM) -> float:
        cache_size = average_n_bits * model.config.num_hidden_layers * model.config.hidden_size
        default_n_bits = torch.finfo(self.dtype).bits
        n_extra = 0 if self.level == "no-quantization" else 1 if self.symmetric else 2
        if self.level == "no-quantization":
            extra_size = 0
        elif self.level == "token":
            extra_size = n_extra * default_n_bits
        elif self.level == "layer":
            extra_size = n_extra * default_n_bits * model.config.num_hidden_layers
        elif self.level == "head":
            extra_size = n_extra * default_n_bits * model.config.num_hidden_layers * model.config.num_attention_heads
        return cache_size + extra_size


def build_quantizers(config_grid_list: list[dict[str, list]]) -> list[Quantizer]:
    quantizer_list: list[Quantizer] = []
    for config_grid in config_grid_list:
        for args in product(*config_grid.values()):
            kwargs = {k: v for k, v in zip(config_grid.keys(), args)}
            quantizer_list.append(Quantizer(**kwargs))
    return quantizer_list
