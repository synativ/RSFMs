import math
from collections import OrderedDict
from collections.abc import Callable

import timm
import torch
from torch import nn

import rsfms.models.decoders as decoder_registry
from rsfms.datasets import HLSBands
from rsfms.models.backbones.convnextv2 import ConvNeXtV2
from rsfms.models.model import (
    AuxiliaryHead,
    AuxiliaryHeadWithDecoderWithoutInstantiatedHead,
    Model,
    ModelFactory,
    register_factory,
)
from rsfms.models.pixel_wise_model import PixelWiseModel


class DecoderNotFoundError(Exception):
    pass


@register_factory
class MMEarthModelFactory(ModelFactory):
    def build_model(
        self,
        task: str,
        backbone: str | nn.Module,
        decoder: str | nn.Module,
        in_channels: int | None = None,
        num_classes: int | None = None,
        pretrained: bool = True,
        num_frames: int = 1,
        prepare_features_for_image_model: Callable | None = None,
        aux_decoders: list[AuxiliaryHead] | None = None,
        rescale: bool = True,
        **kwargs,
    ) -> Model:

        if not torch.cuda.is_available():
            self.CPU_ONLY = True
        else:
            self.CPU_ONLY = False

        # retrieve backbone
        if not isinstance(backbone, nn.Module):
            if not backbone.startswith("mmearth"):
                msg = "This class only handles models for `mmearth` encoders"
                raise NotImplementedError(msg)

            # TO DO: use ConvNeXt V2 in timm to keep this function consistent with the rest of the repo.
            # Current implementation is a workaround to save time to release.
            backbone_dims = kwargs.get('backbone_dims')
            backbone = ConvNeXtV2(
                in_chans=in_channels,
                depths=kwargs.get('backbone_depths'),
                dims=backbone_dims,
                patch_size=kwargs.get('backbone_patch_size'),
                img_size=kwargs.get('backbone_img_size'),
                use_orig_stem=kwargs.get('backbone_use_orig_stem'),
            )

            checkpoint_path = kwargs.get('backbone_checkpoint_path')
            model_dict = torch.load(checkpoint_path)
            backbone_dict = remap_checkpoint_keys(model_dict['model'])
            backbone.load_state_dict(
                backbone_dict, strict=False
            )  # Missing key(s) in state_dict: "norm.weight", "norm.bias", "head.weight", "head.bias".

            # timm implementation
            # backbone_kwargs, kwargs = _extract_prefix_keys(kwargs, "backbone_")
            # backbone: nn.Module = timm.create_model(
            #     backbone,
            #     pretrained=pretrained,
            #     in_chans=in_channels,  # this can be removed, can be derived from bands. But is a breaking change.
            #     num_frames=num_frames,
            #     features_only=True,
            #     **backbone_kwargs,
            # )

        # retrieve decoder
        decoder_cls = _get_decoder(decoder)
        decoder_kwargs, kwargs = _extract_prefix_keys(kwargs, "decoder_")
        # decoder: nn.Module = decoder_cls(backbone.feature_info.channels(), **decoder_kwargs) # timm implementation
        decoder: nn.Module = decoder_cls(backbone_dims, **decoder_kwargs)

        # retrieve head
        head_kwargs, kwargs = _extract_prefix_keys(kwargs, "head_")
        if num_classes:
            head_kwargs["num_classes"] = num_classes

        # retrieve aux decoders
        if aux_decoders is None:
            return PixelWiseModel(
                task, backbone, decoder, head_kwargs, prepare_features_for_image_model, rescale=rescale
            )

        to_be_aux_decoders: list[AuxiliaryHeadWithDecoderWithoutInstantiatedHead] = []
        for aux_decoder in aux_decoders:
            args = aux_decoder.decoder_args if aux_decoder.decoder_args else {}
            aux_decoder_cls: nn.Module = _get_decoder(aux_decoder.decoder)
            aux_decoder_kwargs, kwargs = _extract_prefix_keys(args, "decoder_")
            # aux_decoder_instance = aux_decoder_cls(backbone.feature_info.channels(), **aux_decoder_kwargs) # timm implementation
            aux_decoder_instance = aux_decoder_cls(backbone_dims, **aux_decoder_kwargs)

            aux_head_kwargs, kwargs = _extract_prefix_keys(args, "head_")
            if num_classes:
                aux_head_kwargs["num_classes"] = num_classes
            to_be_aux_decoders.append(
                AuxiliaryHeadWithDecoderWithoutInstantiatedHead(aux_decoder.name, aux_decoder_instance, aux_head_kwargs)
            )

        return PixelWiseModel(
            task,
            backbone,
            decoder,
            head_kwargs,
            prepare_features_for_image_model=prepare_features_for_image_model,
            rescale=rescale,
            auxiliary_heads=to_be_aux_decoders,
        )


def _get_decoder(decoder: str | nn.Module) -> nn.Module:
    if isinstance(decoder, nn.Module):
        return decoder
    if isinstance(decoder, str):
        try:
            decoder = getattr(decoder_registry, decoder)
            return decoder
        except AttributeError as decoder_not_found_exception:
            msg = f"Decoder {decoder} was not found in the registry."
            raise DecoderNotFoundError(msg) from decoder_not_found_exception
    msg = "Decoder must be str or nn.Module"
    raise Exception(msg)


def _extract_prefix_keys(d: dict, prefix: str) -> dict:
    extracted_dict = {}
    remaining_dict = {}
    for k, v in d.items():
        if k.startswith(prefix):
            extracted_dict[k.split(prefix)[1]] = v
        else:
            remaining_dict[k] = v

    return extracted_dict, remaining_dict


def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('encoder'):
            new_k = k.replace('encoder.', '')
            if k.endswith('kernel'):
                new_k = new_k.replace('kernel', 'weight')
                if len(v.shape) == 3:  # reshape standard convolution
                    kv, in_dim, out_dim = v.shape
                    ks = int(math.sqrt(kv))
                    new_ckpt[new_k] = v.permute(2, 1, 0).reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
                elif len(v.shape) == 2:  # reshape depthwise convolution
                    kv, dim = v.shape
                    ks = int(math.sqrt(kv))
                    new_ckpt[new_k] = v.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2)
                continue
            elif 'ln' in k or 'linear' in k:
                new_k = new_k.replace('ln.', '').replace('linear.', '')
            else:
                new_k = new_k
            new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith('bias') and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif 'grn' in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt
