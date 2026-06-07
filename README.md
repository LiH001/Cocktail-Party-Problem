# 鸡尾酒会语音分离实验代码

本项目实现实验指南中的方案 A：基础盲分离。输入是一段单通道双人混合语音，模型输出两路单说话人语音。当前代码主线只保留基于 BLSTM 的深度聚类 / Chimera 风格软掩码分离流程。

本项目只做两组实验：

```text
Easy：简单主实验，3 位说话人，每人 20 条纯净语音，250 段混合样本
Hard：困难对比实验，5 位说话人，每人 30 条纯净语音，300 段混合样本，并加入真实环境噪声
```

Easy 集用于展示模型在较简单条件下的清晰分离效果，目标是让展示样例尽量达到 `SDR_out > 10 dB`。Hard 集用于说明真实噪声和更复杂说话人池会导致 SDR 下降，但模型仍能体现一定分离能力。

## 1. 目录结构

```text
src/
  cocktail_separation/
    audio.py                    # WAV、STFT、ISTFT 等音频工具
    dataset.py                  # JSONL 数据集、特征、理想掩码
    deep_clustering.py          # BLSTM、embedding 分支、软掩码分支、损失函数
    metrics.py                  # SI-SDR 和最优排列评估
  prepare_librispeech_clean.py  # 从 LibriSpeech 整理真实说话人 wav
  build_dataset_manifest.py     # 生成混合语音、参考源和 manifest.jsonl
  split_manifest.py             # 拆分 train.jsonl / valid.jsonl
  train_deep_clustering.py      # 训练 Easy / Hard 模型
  separate_deep_clustering.py   # 对单条混合语音做分离
  generate_deep_clustering_demo.py
  evaluate_separation.py
  evaluate_oracle_masks.py
  debug_overfit_one.py

data/
  raw/
  librispeech_easy_generated/
  librispeech_hard_generated/

outputs/
  logs/
demo_audio_easy/
demo_audio_hard/
gui_app.py
requirements.txt
README.md
```

最终提交包通常包含：

```text
src/
README.md
requirements.txt
demo_audio_easy/
demo_audio_hard/
gui_app.py
report.pdf
```

## 2. 安装依赖

```powershell
pip install -r requirements.txt
```

有 NVIDIA 显卡时建议使用 CUDA 版 PyTorch。训练命令里保留 `--device cuda`；如果电脑没有 GPU，脚本会自动退回 CPU，但训练会慢很多。

## 3. Easy 实验

Easy 是主实验，先把这一组跑通，再做 Hard 对比。

原始语音使用 LibriSpeech 的 `dev-clean.tar.gz`。官方下载页面：

[https://www.openslr.org/12](https://www.openslr.org/12)

直接下载链接：

[https://www.openslr.org/resources/12/dev-clean.tar.gz](https://www.openslr.org/resources/12/dev-clean.tar.gz)

下载后放到：

```text
data/downloads/dev-clean.tar.gz
```

解压到 `data/raw/`：

```powershell
tar -xf data\downloads\dev-clean.tar.gz -C data\raw
```

解压完成后应存在：

```text
data/raw/LibriSpeech/dev-clean
```

Easy 设置：

```text
3 位说话人
每人 20 条真实纯净语音
250 段双说话人混合样本
200 段训练集 + 50 段验证集
混合比例 0.7 : 0.3
采样率 16000 Hz
每段 5 秒
```

如果已经存在下面文件，可以直接跳到训练：

```text
data/librispeech_easy_generated/train.jsonl
data/librispeech_easy_generated/valid.jsonl
```

重新生成 Easy 集：

```powershell
python src\prepare_librispeech_clean.py `
  --librispeech-split data\raw\LibriSpeech\dev-clean `
  --out-dir data\clean_librispeech_easy `
  --speakers 3 `
  --utterances-per-speaker 20 `
  --sample-rate 16000

python src\build_dataset_manifest.py `
  --clean-root data\clean_librispeech_easy `
  --out-dir data\librispeech_easy_generated `
  --num-mixtures 250 `
  --seconds 5 `
  --sample-rate 16000 `
  --source1-gain 0.7 `
  --source2-gain 0.3

python src\split_manifest.py `
  --manifest data\librispeech_easy_generated\manifest.jsonl `
  --train-out data\librispeech_easy_generated\train.jsonl `
  --valid-out data\librispeech_easy_generated\valid.jsonl `
  --valid-ratio 0.2
```

查看 Easy 集的 Oracle 理想掩码上限：

```powershell
python src\evaluate_oracle_masks.py `
  --manifest data\librispeech_easy_generated\valid.jsonl `
  --items 0
```

当前 Easy 集上限约为：

```text
Oracle SDR_in: -0.013 dB
Oracle SDR_out: 12.783 dB
Oracle Delta_SDR: 12.796 dB
```

训练 Easy 模型：

```powershell
python src\train_deep_clustering.py `
  --manifest data\librispeech_easy_generated\train.jsonl `
  --eval-manifest data\librispeech_easy_generated\valid.jsonl `
  --epochs 200 `
  --batch-size 4 `
  --num-speakers 2 `
  --device cuda `
  --dc-loss-weight 0.0 `
  --mask-loss-weight 1.0 `
  --mask-activation sigmoid `
  --eval-every 5 `
  --eval-items 0 `
  --checkpoint outputs\deep_clustering_easy.pt `
  --best-checkpoint outputs\deep_clustering_easy_best.pt `
  --log-dir outputs\logs
```

也可以直接运行默认 Easy 训练：

```powershell
python src\train_deep_clustering.py --device cuda
```

生成 Easy 展示音频：

```powershell
python src\generate_deep_clustering_demo.py `
  --checkpoint outputs\deep_clustering_easy_best.pt `
  --manifest data\librispeech_easy_generated\valid.jsonl `
  --out-dir demo_audio_easy `
  --count 3 `
  --select best
```

`--select best` 会评估整个验证集，并选择 `SDR_out` 最高的 3 条作为展示样例。报告中的整体平均指标应以训练日志或 `demo_audio_easy/metrics_all.csv` 为准，`demo_audio_easy/metrics.csv` 只记录被选出来的展示样例。

## 4. Hard 实验

Hard 是困难对比实验，包含真实环境噪声。

Hard 设置：

```text
5 位说话人
每人 30 条真实纯净语音
300 段双说话人混合样本
240 段训练集 + 60 段验证集
混合比例 0.5 : 0.5
加入真实环境噪声，默认 SNR=10 dB
```

真实噪声推荐使用 ESC-50。下载链接：

[https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip](https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip)

放到：

```text
data/downloads/ESC-50-master.zip
```

解压：

```powershell
tar -xf data\downloads\ESC-50-master.zip -C data\raw
```

重新生成 Hard 集：

```powershell
python src\prepare_librispeech_clean.py `
  --librispeech-split data\raw\LibriSpeech\dev-clean `
  --out-dir data\clean_librispeech_5spk `
  --speakers 5 `
  --utterances-per-speaker 30 `
  --sample-rate 16000

python src\build_dataset_manifest.py `
  --clean-root data\clean_librispeech_5spk `
  --out-dir data\librispeech_hard_generated `
  --num-mixtures 300 `
  --seconds 5 `
  --sample-rate 16000 `
  --source1-gain 0.5 `
  --source2-gain 0.5 `
  --noise-root data\raw\ESC-50-master\audio `
  --snr-db 10

python src\split_manifest.py `
  --manifest data\librispeech_hard_generated\manifest.jsonl `
  --train-out data\librispeech_hard_generated\train.jsonl `
  --valid-out data\librispeech_hard_generated\valid.jsonl `
  --valid-ratio 0.2
```

训练 Hard 模型：

```powershell
python src\train_deep_clustering.py `
  --manifest data\librispeech_hard_generated\train.jsonl `
  --eval-manifest data\librispeech_hard_generated\valid.jsonl `
  --epochs 200 `
  --batch-size 4 `
  --num-speakers 2 `
  --device cuda `
  --dc-loss-weight 0.0 `
  --mask-loss-weight 1.0 `
  --mask-activation sigmoid `
  --eval-every 5 `
  --eval-items 0 `
  --checkpoint outputs\deep_clustering_hard.pt `
  --best-checkpoint outputs\deep_clustering_hard_best.pt `
  --log-dir outputs\logs
```

生成 Hard 展示音频：

```powershell
python src\generate_deep_clustering_demo.py `
  --checkpoint outputs\deep_clustering_hard_best.pt `
  --manifest data\librispeech_hard_generated\valid.jsonl `
  --out-dir demo_audio_hard `
  --count 3 `
  --select best
```

`demo_audio_hard/metrics_all.csv` 记录完整验证集评估结果，`demo_audio_hard/metrics.csv` 只记录被选出来的展示样例。报告中可以说明：Hard 集同时增加说话人数量、使用 0.5:0.5 的更强重叠比例，并加入真实环境噪声，所以 SDR 相比 Easy 集下降是合理现象。

## 5. 评估分离结果

评估 Easy 第一组展示音频：

```powershell
python src\evaluate_separation.py `
  --mixture demo_audio_easy\case_01_deep_clustering\mixture.wav `
  --references demo_audio_easy\case_01_deep_clustering\source_1.wav demo_audio_easy\case_01_deep_clustering\source_2.wav `
  --estimates demo_audio_easy\case_01_deep_clustering\separated_1.wav demo_audio_easy\case_01_deep_clustering\separated_2.wav
```

评估 Hard 第一组展示音频：

```powershell
python src\evaluate_separation.py `
  --mixture demo_audio_hard\case_01_deep_clustering\mixture.wav `
  --references demo_audio_hard\case_01_deep_clustering\source_1.wav demo_audio_hard\case_01_deep_clustering\source_2.wav `
  --estimates demo_audio_hard\case_01_deep_clustering\separated_1.wav demo_audio_hard\case_01_deep_clustering\separated_2.wav
```

指标含义：

```text
SDR_in     分离前混合语音相对参考源的 SI-SDR
SDR_out    分离后语音相对参考源的 SI-SDR
Delta_SDR  SDR_out - SDR_in
```

常用判断：

```text
Delta_SDR > 3 dB      说明分离有明显改善
SDR_out > 5 dB        基本可接受
SDR_out > 10 dB       展示效果较好
```

###  评估完整训练集/验证集（平均、最优与最差数据）

如果想查看 Easy 或 Hard 模型在整个训练集或验证集上的**平均数据**、**最优数据**和**最差数据**，可以通过运行 `generate_deep_clustering_demo.py` 脚本来生成包含所有样本的评估报表。

传入 `--count 0` 参数可以只做评估计算，避免重复写入大量的音频波形文件。

#### (1)评估 Easy 模型在训练集与验证集上的指标：

*   **评估验证集 (Validation Set)**：
    ```powershell
    python src\generate_deep_clustering_demo.py `
      --checkpoint outputs\deep_clustering_easy_best.pt `
      --manifest data\librispeech_easy_generated\valid.jsonl `
      --out-dir outputs\eval_easy_valid `
      --count 0 `
      --select best
    ```

*   **评估训练集 (Training Set)**：
    ```powershell
    python src\generate_deep_clustering_demo.py `
      --checkpoint outputs\deep_clustering_easy_best.pt `
      --manifest data\librispeech_easy_generated\train.jsonl `
      --out-dir outputs\eval_easy_train `
      --count 0 `
      --select best
    ```

#### (2) 评估 Hard 模型在训练集与验证集上的指标：

*   **评估验证集 (Validation Set)**：
    ```powershell
    python src\generate_deep_clustering_demo.py `
      --checkpoint outputs\deep_clustering_hard_best.pt `
      --manifest data\librispeech_hard_generated\valid.jsonl `
      --out-dir outputs\eval_hard_valid `
      --count 0 `
      --select best
    ```

*   **评估训练集 (Training Set)**：
    ```powershell
    python src\generate_deep_clustering_demo.py `
      --checkpoint outputs\deep_clustering_hard_best.pt `
      --manifest data\librispeech_hard_generated\train.jsonl `
      --out-dir outputs\eval_hard_train `
      --count 0 `
      --select best
    ```

#### 3) 如何查看平均、最优与最差数据：

*   **平均数据**：运行上述任一命令后，终端的最后一行会打印该数据集的平均指标，例如：
    `Evaluated summary (60 items): SDR_in=-0.933 dB, SDR_out=1.196 dB, Delta_SDR=2.129 dB`
*   **最优与最差数据**：
    在命令指定的输出目录（如 `outputs\eval_easy_valid\`）下，会生成一个包含全量样本评估分数的 `metrics_all.csv` 文件。
    打开该文件（可用 Excel、WPS 或 Python pandas），以 `sdr_out_db` 或 `delta_sdr_db` 列排序即可查看：
    *   **最优数据**：按数值从大到小（降序）排列，最上面的几行就是性能最好的样本（最优数据）。
    *   **最差数据**：按数值从小到大（升序）排列，最上面的几行就是性能最差的样本（最差数据）。

## 6. GUI 展示界面

项目提供一个 Streamlit GUI 入口 `gui_app.py`，用于展示 demo 音频、指标 CSV、训练日志和 checkpoint 分离效果。

启动方式：

```powershell
python -m streamlit run gui_app.py
```

启动后浏览器会打开本地页面；如果没有自动打开，可以访问终端中显示的地址，通常是：

```text
http://localhost:8501
```

停止 GUI：

```text
在运行 Streamlit 的终端窗口按 Ctrl + C
```

如果 GUI 是后台启动的，或者已经找不到原来的终端窗口，可以在 PowerShell 中查看占用 8501 端口的进程：

```powershell
netstat -ano | findstr :8501
```

找到最后一列的 PID 后结束进程，例如 PID 是 `12345`：

```powershell
taskkill /PID 12345 /F
```

GUI 包含四个页面：

```text
Demo Compare              选择 Easy / Hard 和 case，播放 mixture、source、separated，并查看 SDR 指标
Waveform And Spectrogram  查看混合语音、参考源、分离结果和噪声的波形与频谱图
Custom Separation         选择已有 checkpoint 和混合 WAV，或上传 WAV，生成两路分离音频
Metrics Overview          汇总 demo 指标、完整评估指标和 outputs/logs 中的训练曲线
```

自定义分离页面会把上传音频暂存到：

```text
outputs/gui_uploads/
```

分离结果会保存到：

```text
outputs/gui_separated/
```

Hard 样例中的噪声展示分为两种：`Noise Reference` 是数据生成时使用的噪声素材或 demo 中保存的噪声片段；`Estimated Noise In Mixture` 是 GUI 按 `mixture - source_1 - source_2` 估算出的混合语音中实际噪声成分，会保存到：

```text
outputs/gui_noise_estimates/
```

如果 GUI 中没有显示 checkpoint，请先确认 `outputs/` 下存在 `*.pt` 模型文件，例如：

```text
outputs/deep_clustering_easy_best.pt
outputs/deep_clustering_hard_best.pt
```

GUI 展示流程建议为：先听混合语音，再听分离结果，同时用波形、频谱和 SDR 指标说明 Easy 与 Hard 实验的差异。

## 7. 单条混合语音分离

以 Easy 中一条混合语音为例：

```powershell
python src\separate_deep_clustering.py `
  --checkpoint outputs\deep_clustering_easy_best.pt `
  --mixture data\librispeech_easy_generated\mix\mix_000_speaker_2277_speaker_3752.wav `
  --out-dir outputs\separated `
  --num-speakers 2
```

然后评估：

```powershell
python src\evaluate_separation.py `
  --mixture data\librispeech_easy_generated\mix\mix_000_speaker_2277_speaker_3752.wav `
  --references data\librispeech_easy_generated\sources\mix_000_speaker_2277_speaker_3752_s1.wav data\librispeech_easy_generated\sources\mix_000_speaker_2277_speaker_3752_s2.wav `
  --estimates outputs\separated\speaker_1.wav outputs\separated\speaker_2.wav
```

如果文件名不存在，请到对应数据集的 `mix/` 中选择实际存在的 `mix_*.wav`，再到 `sources/` 中选择同名的 `_s1.wav` 和 `_s2.wav`。如果 `outputs\separated\speaker_1.wav` 不存在，说明还没有先运行分离脚本生成这两个文件。

## 8. 训练输出说明

训练结束后会输出最佳模型摘要：

```text
Best validation summary:
  epoch: ...
  SDR_in: ... dB
  SDR_out: ... dB
  Delta_SDR: ... dB
  checkpoint: ...
```

每次训练还会保存一份 CSV 日志到 `outputs/logs/`，文件名类似：

```text
deep_clustering_easy_20260604_153000.csv
deep_clustering_hard_20260604_180000.csv
```

日志中每个 epoch 一行，包含 `mean_loss`、`dc_loss`、`sa_loss`、`SDR_in`、`SDR_out`、`Delta_SDR` 和是否刷新最佳模型，后续写报告或画曲线可以直接读取。

参数说明：

```text
dc-loss-weight 0.0      主训练使用 PIT 软掩码分离损失，更容易稳定提升 SDR
mask-loss-weight 1.0    用分离后幅度谱逼近真实源幅度谱
mask-activation sigmoid 两个输出掩码独立预测
eval-items 0            每次评估使用完整验证集
log-dir                 每次训练保存一份 CSV 日志，默认在 outputs/logs
```

如果训练结果异常偏低，可以先做单样本过拟合测试：

```powershell
python src\debug_overfit_one.py `
  --manifest data\librispeech_easy_generated\train.jsonl `
  --index 0 `
  --steps 500 `
  --device cuda `
  --out-dir outputs\overfit_one
```

单样本能达到较高 SDR，说明模型、损失函数和评估链路是通的；验证集偏低通常是泛化不足或数据难度太高。

