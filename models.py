import math
from torch import nn
import cv2


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4): # d : dimension, m : mapping layers 수
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            # Feature extraction (num_channels -> d)
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        # 두 개의 convolution 사이에 사용되어 연결 수(파라미터)를 줄임 => 모델 크기 감소 => 네트워크 속도 향상 => SRCNN보다 빠름
        # Shrinking (d -> s)
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        
        for _ in range(m):
            # Mapping (s -> s)
            # 일관성을 위해 모든 mapping layers는 같은 filter 수를 가짐
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        
        # Expanding (shrinking layer와 정반대 process)
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        
        # Deconvolution (=> upsampling)
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True, dilation=1, padding_mode='zeros')
        # SRCNN의 first layer와 같은 구성의 9x9 filter를 사용
        # deconvolution filter의 패턴이 SRCNN의 first layer와 매우 비슷함
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)
    
    # # Tensor -> numpy
    # def to_np(t):
    #     return t.cpu().detach().numpy()
    
    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        # y = to_np(x)
        # img = cv2.imread()
        x = self.last_part(x)
        # y2 = to_np(x)
        return x


