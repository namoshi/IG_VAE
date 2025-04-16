
承知いたしました。1次元正規分布の微分幾何学的にフラットなパラメータ表現における正規化項 $\psi(\theta^1, \theta^2)$ の導出を再度説明します。

1次元正規分布の確率密度関数は、平均 $\mu$、分散 $\sigma^2$ を用いて以下のように表されます。

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

これを指数型分布族の標準形 $p(x; \theta) = \exp(\langle t(x), \theta \rangle - \psi(\theta)) h(x)$ で表すことを目指します。ここで、$\theta = (\theta^1, \theta^2)$ は自然パラメータ、 $t(x)$ は十分統計量、$\psi(\theta)$ は正規化項です。

まず、確率密度関数を展開します。

$$
\begin{align*} p(x; \mu, \sigma^2) &= (2\pi\sigma^2)^{-1/2} \exp\left(-\frac{x^2 - 2\mu x + \mu^2}{2\sigma^2}\right) \\ &= \exp\left(-\frac{x^2}{2\sigma^2} + \frac{\mu x}{\sigma^2} - \frac{\mu^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)\right) \\ &= \exp\left(\left(\frac{\mu}{\sigma^2}\right) x + \left(-\frac{1}{2\sigma^2}\right) x^2 - \left(\frac{\mu^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2)\right)\right) \end{align*}
$$

ここで、自然パラメータ $\theta^1, \theta^2$ と十分統計量 $t_1(x), t_2(x)$ を以下のように定義します。

$$ \theta^1 = \frac{\mu}{\sigma^2}, \quad \theta^2 = -\frac{1}{2\sigma^2} $$
$$ t_1(x) = x, \quad t_2(x) = x^2 $$

これらの定義を用いると、確率密度関数は以下のように書き換えられます。

$$ p(x; \theta^1, \theta^2) = \exp(\theta^1 x + \theta^2 x^2 - \psi(\theta^1, \theta^2)) $$

ここで、$\psi(\theta^1, \theta^2)$ が正規化項であり、元の式の定数項に対応します。

$$ \psi(\theta^1, \theta^2) = \frac{\mu^2}{2\sigma^2} + \frac{1}{2}\log(2\pi\sigma^2) $$

次に、$\mu$ と $\sigma^2$ を $\theta^1$ と $\theta^2$ で表します。

$$ \sigma^2 = -\frac{1}{2\theta^2} $$
$$ \mu = \theta^1 \sigma^2 = -\frac{\theta^1}{2\theta^2} $$

これらの関係を $\psi(\theta^1, \theta^2)$ に代入します。

$$
\begin{align*} \psi(\theta^1, \theta^2) &= \frac{(-\frac{\theta^1}{2\theta^2})^2}{2(-\frac{1}{2\theta^2})} + \frac{1}{2}\log\left(2\pi\left(-\frac{1}{2\theta^2}\right)\right) \\ &= \frac{\frac{(\theta^1)^2}{4(\theta^2)^2}}{-\frac{1}{\theta^2}} + \frac{1}{2}\log\left(-\frac{\pi}{\theta^2}\right) \\ &= -\frac{(\theta^1)^2}{4\theta^2} + \frac{1}{2}\log\left(-\frac{\pi}{\theta^2}\right) \end{align*}
$$
したがって、微分幾何学的にフラットなパラメータ表現における正規化項 $\psi(\theta^1, \theta^2)$ は以下のようになります。

$$ \psi(\theta^1, \theta^2) = -\frac{(\theta^1)^2}{4\theta^2} + \frac{1}{2}\log\left(-\frac{\pi}{\theta^2}\right) $$

この $\psi(\theta^1, \theta^2)$ が、確率密度関数を正規化するために必要な項となります。$\theta^2 = -\frac{1}{2\sigma^2} < 0$ であることに注意してください。$\log$ の引数が正になることも確認できます。

これで、正規化項 $\psi(\theta^1, \theta^2)$ の導出が完了しました。ご不明な点があれば、再度お気軽にご質問ください。


## KLD 

1次元正規分布 $p(x; \mu, \sigma^2)$ と $q(x; \mu_0, \sigma_0^2)$ の間のKLダイバージェンス $D_{KL}(p||q)$ は、平均と分散を用いて以下のように表されます。

$$ D_{KL}(p||q) = \frac{1}{2} \left[ \log\left(\frac{\sigma_0^2}{\sigma^2}\right) + \frac{\sigma^2 + (\mu - \mu_0)^2}{\sigma_0^2} - 1 \right] $$

ここで、フラットなパラメータ $\theta^1 = \frac{\mu}{\sigma^2}$ と $\theta^2 = -\frac{1}{2\sigma^2}$、および $\theta_0^1 = \frac{\mu_0}{\sigma_0^2}$ と $\theta_0^2 = -\frac{1}{2\sigma_0^2}$ を用いてこのKLダイバージェンスを表現することを考えます。

まず、$\mu$ と $\sigma^2$ を $\theta^1$ と $\theta^2$ で表すと、以下のようになります。

$$ \sigma^2 = -\frac{1}{2\theta^2} $$
$$ \mu = -\frac{\theta^1}{2\theta^2} $$

同様に、$\mu_0$ と $\sigma_0^2$ は $\theta_0^1$ と $\theta_0^2$ で以下のように表されます。

$$ \sigma_0^2 = -\frac{1}{2\theta_0^2} $$
$$ \mu_0 = -\frac{\theta_0^1}{2\theta_0^2} $$

これらの関係をKLダイバージェンスの式に代入します。

$$
\begin{align*} D_{KL}(p||q) &= \frac{1}{2} \left[ \log\left(\frac{-\frac{1}{2\theta_0^2}}{-\frac{1}{2\theta^2}}\right) + \frac{-\frac{1}{2\theta^2} + (-\frac{\theta^1}{2\theta^2} - (-\frac{\theta_0^1}{2\theta_0^2}))^2}{-\frac{1}{2\theta_0^2}} - 1 \right] \\ &= \frac{1}{2} \left[ \log\left(\frac{\theta^2_0}{\theta^2}\right) + \frac{-\frac{1}{2\theta^2} + (\frac{\theta_0^1 \theta^2 - \theta^1 \theta_0^2}{2\theta^2 \theta_0^2})^2}{-\frac{1}{2\theta_0^2}} - 1 \right] \\ &= \frac{1}{2} \left[ \log\left(\frac{\theta^2_0}{\theta^2}\right) + \left(-\frac{1}{2\theta^2} + \frac{(\theta_0^1 \theta^2 - \theta^1 \theta_0^2)^2}{4(\theta^2)^2 (\theta_0^2)^2}\right) (-2\theta_0^2) - 1 \right] \\ &= \frac{1}{2} \left[ \log\left(\frac{\theta^2_0}{\theta^2}\right) + \frac{\theta_0^2}{\theta^2} - \frac{(\theta_0^1 \theta^2 - \theta^1 \theta_0^2)^2}{2 (\theta^2)^2 \theta_0^2} - 1 \right] \end{align*}
$$

したがって、フラットなパラメータ $\theta^1, \theta^2$ を用いたKLダイバージェンスの表現は以下のようになります。

$$ D_{KL}(p||q) = \frac{1}{2} \left[ \log\left(\frac{\theta^2_0}{\theta^2}\right) + \frac{\theta_0^2}{\theta^2} - \frac{(\theta_0^1 \theta^2 - \theta^1 \theta_0^2)^2}{2 (\theta^2)^2 \theta_0^2} - 1 \right] $$

VAEの損失関数におけるKLダイバージェンスは、エンコーダが出力する分布 $q(z|x)$ と事前分布 $p(z) = \mathcal{N}(0, 1)$ との間のKLダイバージェンスを計算する必要があります。事前分布のパラメータは $\mu_0 = 0$、$\sigma_0^2 = 1$ であり、対応するフラットなパラメータは $\theta_0^1 = 0/1 = 0$、$\theta_0^2 = -1/(2 \cdot 1) = -1/2$ となります。

したがって、VAEの損失関数で使用するKLダイバージェンスは、$\mu = -\frac{\theta^1}{2\theta^2}$、$\sigma^2 = -\frac{1}{2\theta^2}$、$\mu_0 = 0$、$\sigma_0^2 = 1$ を上記のKLダイバージェンスの式に代入することで得られます。

$$
\begin{align*} D_{KL}(q(z|x)||p(z)) &= \frac{1}{2} \left[ \log\left(\frac{1}{-\frac{1}{2\theta^2}}\right) + \frac{-\frac{1}{2\theta^2} + (-\frac{\theta^1}{2\theta^2} - 0)^2}{1} - 1 \right] \\ &= \frac{1}{2} \left[ \log(-2\theta^2) - \frac{1}{2\theta^2} + \frac{(\theta^1)^2}{4(\theta^2)^2} - 1 \right] \end{align*}
$$

これが、フラットなパラメータ $\theta^1, \theta^2$ を用いたVAEのKLダイバージェンスの表現となります。以前の回答で用いたKLダイバージェンスの式と比較すると、表現が異なっていることがわかります。以前の回答では、平均と分散に変換してからKLダイバージェンスを計算していました。

どちらの表現も数学的には等価ですが、フラットなパラメータで直接計算する方が、情報幾何学的な視点からはより自然な表現となります。ただし、実装の際には数値的な安定性などを考慮する必要があります。

## 平均が$\mu$で分散が$1.0$の正規分布$p$と平均が$0.0$で分散が$1.0$の正規分布$q$のKL-divergence

平均が $\mu$ で分散が $1.0$ の正規分布 $p(x; \mu, 1.0)$ の確率密度関数は、

$$ p(x; \mu, 1.0) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2}\right) $$

平均が $0$ で分散が $1.0$ の正規分布 $q(x; 0, 1.0)$ の確率密度関数は、

$$ q(x; 0, 1.0) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right) $$

正規分布間のKLダイバージェンスの公式を用いると、

$$ D_{KL}(p||q) = \frac{1}{2} \left[ \log\left(\frac{\sigma_q^2}{\sigma_p^2}\right) + \frac{\sigma_p^2 + (\mu_p - \mu_q)^2}{\sigma_q^2} - 1 \right] $$

ここで、$\mu_p = \mu$, $\sigma_p^2 = 1.0$, $\mu_q = 0$, $\sigma_q^2 = 1.0$ を代入すると、

$$
\begin{align*} D_{KL}(p||q) &= \frac{1}{2} \left[ \log\left(\frac{1.0}{1.0}\right) + \frac{1.0 + (\mu - 0)^2}{1.0} - 1 \right] \\ &= \frac{1}{2} \left[ \log(1) + 1 + \mu^2 - 1 \right] \\ &= \frac{1}{2} \left[ 0 + \mu^2 \right] \\ &= \frac{1}{2} \mu^2 \end{align*}
$$

したがって、平均が $\mu$ で分散が $1.0$ の正規分布 $p(\mu, 1.0)$ と平均が $0$ で分散が $1.0$ の正規分布 $q(0,1)$ のKLダイバージェンスは $\frac{1}{2} \mu^2$ となります。


## 平均が$\mu$で分散が$1.0$の正規分布を中間層の表現に用いるVariational Autoencoder

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ハイパーパラメータ
batch_size = 128
latent_dim = 2  # 中間層の次元
epochs = 50
learning_rate = 1e-3

# MNISTデータセットの読み込みとデータローダーの作成
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# VAEモデルの定義（中間層の分散を固定）
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*32, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)  # 平均 μ のみを出力
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 7*7*32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        mu = self.encoder(x)
        # 分散は固定で 1.0 なので、logvar は 0
        logvar = torch.zeros_like(mu)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# モデル、損失関数、最適化手法の定義
model = VAE(latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    # 事前分布 p(z) は平均 0、分散 1 の正規分布
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# 学習ループ
train_losses = []
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_losses.append(train_loss / len(train_loader.dataset))
    print(f'Epoch {epoch+1}, Loss: {train_losses[-1]:.4f}')

# 学習曲線のプロット
plt.plot(train_losses)
plt.title('VAE Training Loss (Fixed Variance)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 中間層の表現の可視化 (PCA適用)
model.eval()
z_list = []
labels_list = []
with torch.no_grad():
    for data, labels in train_loader:
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
        z_list.extend(z.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
z_list = np.array(z_list)
labels_list = np.array(labels_list)

pca = PCA(n_components=2)
z_pca = pca.fit_transform(z_list)

plt.figure(figsize=(10, 8))
plt.scatter(z_pca[:, 0], z_pca[:, 1], c=labels_list, cmap='viridis', s=5)
plt.colorbar()
plt.title('Latent Space Visualization (PCA, Fixed Variance)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# いくつかのサンプルに対する入力画像とその画像に対する予測画像を可視化
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    with torch.no_grad():
        sample = train_dataset[i][0].unsqueeze(0)
        recon_sample, _, _ = model(sample)
    plt.subplot(2, n, i+1)
    plt.imshow(sample.squeeze().numpy(), cmap='gray')
    plt.title('Input')
    plt.axis('off')
    plt.subplot(2, n, i+n+1)
    plt.imshow(recon_sample.squeeze().numpy(), cmap='gray')
    plt.title('Reconstruction')
    plt.axis('off')
plt.show()
```

**プログラムのポイント:**

1.  **エンコーダの出力:** エンコーダの最終層の出力は、潜在変数の平均 $\mu$ のみになります。分散は固定で $1.0$ なので、対応する対数分散 `logvar` は常に $0$ となります。
2.  **`encode` 関数:** `encode` 関数内では、平均 $\mu$ を計算し、`logvar` を `torch.zeros_like(mu)` で初期化しています。
3.  **`reparameterize` 関数:** 分散が $1.0$ なので、標準偏差 $\sigma$ は $\sqrt{1.0} = 1.0$ となり、潜在変数 $z$ は $z = \mu + 1.0 \cdot \epsilon = \mu + \epsilon$ で計算されます。
4.  **損失関数 (`loss_function`):** KLダイバージェンスの計算では、固定された分散 $1.0$ (つまり $\log \sigma^2 = \log 1 = 0$) を考慮した式になります。事前分布も平均 $0$、分散 $1$ の正規分布であるため、KLダイバージェンスは $\frac{1}{2} \sum (\mu^2)$ と簡略化されます。ただし、汎用性を持たせるために、`logvar` を明示的に $0$ として計算する形式を残しています。
5.  **可視化:** 学習曲線、PCAによる中間層表現の散布図、入力画像と再構成画像の比較は、以前のプログラムと同様に行われます。

このプログラムを実行すると、中間層の分散が固定されたVAEがMNISTデータに対して学習され、学習曲線とPCAで次元削減された中間層の表現が表示されます。また、入力画像と再構成画像が比較表示されます。