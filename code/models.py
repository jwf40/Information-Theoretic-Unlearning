"""
From https://github.com/vikram2000b/bad-teaching-unlearning
And https://github.com/weiaicunzai/pytorch-cifar100 (better performance) <- Refer to this for comments
"""

from torch import nn
import numpy as np
import torch
from torchvision.models import resnet18, vgg11, vgg16
from transformers import ViTModel, ViTFeatureExtractor
from resnet import ResNet, BasicBlock


class VGG16(nn.Module):
    def __init__(self, num_classes=10, return_activations=False):
        super(VGG16, self).__init__()
        self.return_activations = return_activations
        self.model = vgg11()
        self.fc=nn.Linear(1000, num_classes)

    def forward(self, x):      
        if not self.return_activations:
            x = self.model(x)
            return self.fc(x)
            #return self.model(x)       

        activation_list = []
        for layer in self.model.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and (x.numel() > 0):
                activation_list.append(x)        
        
        x = self.model.avgpool(x)        
        x = torch.flatten(x,1)
        for layer in self.model.classifier:
            x = layer(x)
            if isinstance(layer, nn.Linear) and (x.numel()> 0):
                activation_list.append(x)
        #x = self.model.classifier(x)       

        x = self.fc(x)
        
        return x, activation_list

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvStandard(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        output_padding=0,
        w_sig=np.sqrt(1.0),
    ):
        super(ConvStandard, self).__init__(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.w_sig = w_sig
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(
            self.weight,
            mean=0,
            std=self.w_sig / (self.in_channels * np.prod(self.kernel_size)),
        )
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        output_padding=0,
        activation_fn=nn.ReLU,
        batch_norm=True,
        transpose=False,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            #             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            #                                 )]
            model += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=not batch_norm,
                )
            ]
        else:
            model += [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=not batch_norm,
                )
            ]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN(nn.Module):
    def __init__(
        self,
        filters_percentage=1.0,
        n_channels=3,
        num_classes=10,
        dropout=False,
        batch_norm=True,
    ):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(
                n_filter1,
                n_filter2,
                kernel_size=3,
                stride=2,
                padding=1,
                batch_norm=batch_norm,
            ),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(
                n_filter2,
                n_filter2,
                kernel_size=3,
                stride=2,
                padding=1,
                batch_norm=batch_norm,
            ),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output


class ViT(nn.Module):
    def __init__(self, num_classes=20, return_activations=False, **kwargs):
        super(ViT, self).__init__()
        self.return_activations = return_activations
        self.base = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.final = nn.Linear(self.base.config.hidden_size, num_classes)
        self.num_classes = num_classes
        self.relu = nn.ReLU()

    def forward(self, pixel_values):
        if not self.return_activations:
            outputs = self.base(pixel_values=pixel_values)
            logits = self.final(outputs.last_hidden_state[:, 0])
            return logits
        
        activations = []
        # x = self.base.embeddings(pixel_values)
        # activations.append(x)
        # x = self.base.encoder(x)
        # activations.append(x)
        # x = self.base.layernorm(x)
        # x = self.base.pooler(x)        
        # logits = self.final(x.last_hidden_state[:, 0])

        

        output_attentions = self.base.config.output_attentions
        output_hidden_states = (
            self.base.config.output_hidden_states
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.base.get_head_mask(None, self.base.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.base.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype) 
        
        embedding_output = self.base.embeddings(
            pixel_values, bool_masked_pos=None, interpolate_pos_encoding=None
        )

        for i, layer_module in enumerate(self.base.encoder.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (embedding_output,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(embedding_output, layer_head_mask, output_attentions)
            if layer_outputs[0].numel() > 0:
                activations.append(layer_outputs[0])
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        last_hidden_state=hidden_states
        
        sequence_output = last_hidden_state
        sequence_output = self.base.layernorm(sequence_output)

        last_hidden_state=sequence_output

        logits = self.final(last_hidden_state[:, 0])       

        return logits, activations