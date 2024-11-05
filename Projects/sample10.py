# ラベルを付与するやり方をGPTに書いてもらって、それを動くように書き換えたやつ
# ラベルの次元そんなたくさんある必要なくね？って思って、とりあえずベクトルを16*16で用意して1つのチャンネルにまとめた
# エンコーダにいれる画像は一定じゃないと意味なくない？と思うから、とりあえず1枚目だけで学習してみる

from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import cv2
import ffmpeg

image_path = 'data/misty_64_64.avi'
basename = os.path.basename(image_path)
num_epochs = 1000
image_size = 64

num_bits = 8
hidden_layer_channels = 16
encoder_output_channels = 8
num_labels = 64
embedding_dim = pow(image_size//4, 2)  # 埋め込みベクトルの次元

train_model = True
save_model = True

project_name = "sample10"

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# フルカラー画像用エンコーダの定義
class ColorEncoder(nn.Module):
    def __init__(self):
        super(ColorEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, encoder_output_channels, 3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

# フルカラー画像用デコーダの定義
class ColorDecoder(nn.Module):
    def __init__(self):
        super(ColorDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_output_channels + 1, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

# 量子化関数の定義
def quantize(tensor, num_bits):
    scale = pow(2, num_bits) - 1
    return (torch.round(tensor * scale) / scale)

# ファイルパスを取得しndarrayを返す
def readClip(filepass): 
    cap = cv2.VideoCapture(filepass)
    print(cap.isOpened())# 読み込めたかどうか
    ret,frame = cap.read()
    array = np.reshape(frame,(1,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3))
    while True: 
        ret, frame = cap.read()# 1フレーム読み込み
        if ret == False :# フレームが読み込めなかったら出る
            break
        frame = np.reshape(frame,(1,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3))       
        array = np.append(array,frame,axis=0)
    cap.release()
    return array

def timelaps(movie, saved_name, all_frame=64, width=64, height=64, frame_rate=32):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(saved_name, fourcc, frame_rate, (width, height))
    
    for i in range(all_frame):
        img = movie[i]
        video.write(img) 
        
    video.release()
    print("動画変換完了")
    
# モデルのインスタンス化
encoder = ColorEncoder().to(device)
decoder = ColorDecoder().to(device)
label_embedding = nn.Embedding(num_labels, embedding_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(label_embedding.parameters()), lr=1e-3)

# 学習・圧縮・復元の処理
def process_images(images, train_model, save_model):
    save_name = f"{project_name}_{device}_{num_epochs}_{num_bits}"
    start = time.perf_counter()
    if train_model:
        # 単一の画像での訓練ループ
        for epoch in range(num_epochs):
            for label_index in range(64):
                label_emb = label_embedding(torch.tensor([label_index], dtype=torch.long).to(device))
                label_emb = label_emb.view(1, 1, image_size//4, image_size//4)

                # 順伝播
                encoder_output = encoder(image[:, :, 0])

                if epoch < num_epochs * 0.95:
                    # 量子化誤差を考慮した一様分布ノイズを生成
                    noise = (torch.rand_like(encoder_output) - 0.5) / (2 ** num_bits)
                    decoder_input = encoder_output + noise
                else:
                    decoder_input = quantize(encoder_output, num_bits)

                # print(decoder_input.shape, label_emb.shape)

                decoder_input = torch.cat((decoder_input, label_emb), dim=1)

                decoder_output = decoder(decoder_input)
                loss = criterion(decoder_output, image[:, :, label_index])

                # 逆伝播と最適化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        end = time.perf_counter()
        print("学習時間：" + str(end - start))

        # エンコーダとデコーダの保存
        torch.save(encoder.state_dict(), f'model/{save_name}_encoder.pth')
        torch.save(decoder.state_dict(), f'model/{save_name}_decoder.pth')
        torch.save(label_embedding.state_dict(), f'model/{save_name}_embedding.pth')

    else:
        # 訓練済みパラメータのロード
        decoder.load_state_dict(torch.load(f'model/{save_name}_decoder.pth'))
        label_embedding.load_state_dict(torch.load(f'model/{save_name}_embedding.pth'))
        decoder.eval()
        label_embedding.eval()

    if save_model:
        if not train_model:
            # エンコーダの訓練済みパラメータのロード
            encoder.load_state_dict(torch.load(f'model/{save_name}_encoder.pth'))
            encoder.eval()

        # エンコード
        start = time.perf_counter()
        with torch.no_grad():
            encoder_outputs = []
            selected_image = image[:,:,0]
            encoder_output = encoder(selected_image)

            decoder_input = quantize(encoder_output, num_bits)
            compressed = (decoder_input * (pow(2, num_bits) - 1)).to(torch.uint8)

        end = time.perf_counter()
        print("圧縮時間：" + str(end - start))

        # 圧縮された潜在表現を保存
        compressed_np = compressed.cpu().numpy()
        np.save(f'comp/{save_name}.npy', compressed_np)

    else:
        # 圧縮された潜在表現を読み込み
        compressed_np = np.load(f'comp/{save_name}.npy')
        compressed = torch.tensor(compressed_np).to(device)
        decoder_input = compressed.to(torch.float32) / (pow(2, num_bits) - 1)

    # デコード
    start = time.perf_counter()
    with torch.no_grad():
        reconstructed = torch.empty_like(image)
        for label_index in range(64):
            label_emb = label_embedding(torch.tensor([label_index], dtype=torch.long).to(device))
            label_emb = label_emb.view(1, 1, image_size//4, image_size//4)

            decoder_input_with_label = torch.cat((decoder_input, label_emb), dim=1)
            decoder_output = decoder(decoder_input_with_label)
            reconstructed[:,:,label_index] = decoder_output
    end = time.perf_counter()
    print("解凍時間：" + str(end - start))

    reconstructed_image = reconstructed.squeeze().permute(1, 2, 3, 0).cpu().numpy() * 255 # (64, 64, 64, 3)
    reconstructed_image = reconstructed_image.astype(np.uint8)
    # reconstructed_image = Image.fromarray(reconstructed_image)

    timelaps(reconstructed_image, f'image/{save_name}.avi')


    # 画像の表示
    plt.figure(figsize=(6, 3))

    # オリジナル画像
    plt.subplot(1, 2, 1)
    plt.imshow(image.cpu().numpy().squeeze().transpose(1, 2, 3, 0)[0])
    plt.axis('off')
    plt.title('Original Image')

    # 再構成された画像
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.cpu().numpy().squeeze().transpose(1, 2, 3, 0)[0])
    plt.axis('off')
    plt.title('Reconstructed Image')

    plt.show()

# 画像を読み込む
image = readClip(image_path)

# numpy 配列を torch.Tensor に変換し、サイズを変更
image = torch.tensor(image).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0  # [T, H, W, C] -> [C, T, H, W] -> [1, C, T, H, W]
image = image.to(device)

# 画像の処理
process_images(image, train_model, save_model)
