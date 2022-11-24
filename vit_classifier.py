import torch
from einops import rearrange
from torch import nn, optim

from utils import EarlyStopping, train_model


def img_to_patches(img, patch_size=16, patch_stride=16):
    batch_size, C, W, H = img.shape
    img = img.permute((0, 2, 3, 1))
    patches = img.unfold(1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)
    patches = patches.reshape((batch_size, (W // patch_size) * (H // patch_stride), -1))
    return patches


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ViT_Encoder_Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super(ViT_Encoder_Block, self).__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.MHA = Attention(d_model, heads=num_heads)

        # self.MHA = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        # self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffd1 = nn.Linear(d_model, d_model * 4)
        self.gelu = nn.GELU()
        self.ffd2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        out1 = self.layer_norm1(x)

        # q, k, v = self.to_qkv(out1).chunk(3, dim=-1)
        # attn_output, _ = self.MHA(q, k, v)
        attn_output = self.MHA(out1)

        out1 = attn_output + x
        out2 = self.layer_norm2(out1)
        out2 = self.ffd1(out2)
        out2 = self.gelu(out2)
        out2 = self.ffd2(out2)
        return out1 + out2


class ViT_Classifier(nn.Module):
    def __init__(self, num_classes, channels=3, img_width=224, img_height=224,
                 patch_size=16, patch_stride=16,
                 d_model=768, num_layers=8, num_heads=12):
        super(ViT_Classifier, self).__init__()
        self.channels = channels
        self.img_width = img_width
        self.img_height = img_height
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.cls_token = nn.Parameter(torch.randn(d_model))
        self.PEs = nn.Parameter(torch.randn((img_width // patch_size) * (img_height // patch_stride) + 1, d_model))
        self.fc = nn.Linear(channels * patch_size * patch_stride, d_model)
        self.layers = nn.Sequential(*[ViT_Encoder_Block(d_model, num_heads) for _ in range(num_layers)])
        self.end_FC = nn.Linear(d_model, num_classes)

    def forward(self, img):
        x = img_to_patches(img, self.patch_size, self.patch_stride)
        x = self.fc(x)
        batch_sz = img.shape[0]
        token = self.cls_token.repeat(batch_sz, 1, 1)
        x_embed = torch.cat((token, x), dim=1)
        pes = self.PEs.repeat(batch_sz, 1, 1)
        x_embed_pos = x_embed + pes
        out = self.layers(x_embed_pos)
        out = self.end_FC(out[:, 0])
        return out


def get_vit_model(num_classes, num_layers, device):
    return ViT_Classifier(num_classes=num_classes, num_layers=num_layers).to(device)


def train_vit(dataloaders, image_datasets, num_classes, device, num_epochs=50):
    model = get_vit_model(num_classes, num_layers=8, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    early_stopping = EarlyStopping(model_name="VisionTransformer", save_best=True)
    torch.cuda.empty_cache()
    return train_model(model, dataloaders, image_datasets, criterion,
                       optimizer, device, early_stopping, num_epochs)
