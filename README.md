# 将棋盤検出プロジェクト

このプロジェクトは、画像から将棋盤を検出し、81 マスに分割して保存する Python アプリケーションです。

## セットアップ

### 1. 仮想環境の作成とアクティベート

```bash
# 仮想環境を作成
python3 -m venv venv

# 仮想環境をアクティベート
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate     # Windows
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 使用方法

1. 将棋盤の画像ファイルをプロジェクトディレクトリに配置
2. `app.py`の`input_image_file`変数を画像ファイル名に変更
3. スクリプトを実行

```bash
python app.py
```

## 仮想環境の管理

### 仮想環境をアクティベート

```bash
source venv/bin/activate
```

### 仮想環境を非アクティベート

```bash
deactivate
```

### 仮想環境を削除

```bash
rm -rf venv
```

## 依存関係

- opencv-python: 画像処理ライブラリ
- numpy: 数値計算ライブラリ

## 出力

処理が完了すると、`output_cells`ディレクトリに 81 個の画像ファイル（`cell_0_0.png`から`cell_8_8.png`）が生成されます。
