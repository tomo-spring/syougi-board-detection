import cv2
import numpy as np
import os
import math

def _cluster_lines(lines, axis, threshold=10):
    """
    近くにある直線をクラスタリングして、1本の直線にまとめる
    axis=0: 垂直な直線 (x座標でクラスタリング)
    axis=1: 水平な直線 (y座標でクラスタリング)
    """
    if lines is None or len(lines) == 0:
        return []
    
    lines.sort()
    
    clusters = []
    current_cluster = [lines[0]]
    
    for i in range(1, len(lines)):
        # 閾値より近ければ同じクラスタに追加
        if lines[i] - current_cluster[-1] < threshold:
            current_cluster.append(lines[i])
        # 遠ければ新しいクラスタを作成
        else:
            clusters.append(int(np.mean(current_cluster)))
            current_cluster = [lines[i]]
    
    # 最後のクラスタを追加
    clusters.append(int(np.mean(current_cluster)))
    
    return clusters

def detect_shogi_grid_and_crop(image_path, output_dir='output_cells_grid'):
    """
    将棋盤の格子線を検出して81マスに分割し、保存する関数

    Args:
        image_path (str): 入力画像のパス
        output_dir (str): 切り抜いた画像を保存するディレクトリ名
    """
    # === ステップ1: 将棋盤の大枠の検出（前回と同様） ===
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    if img is None:
        print(f"エラー: 画像 '{image_path}' を読み込めません。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 10, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    board_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                max_area = area
                board_contour = approx

    if board_contour is None:
        print("将棋盤の輪郭を検出できませんでした。")
        return

    points = board_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    side_length = 540
    dst_points = np.array([[0, 0], [side_length - 1, 0], [side_length - 1, side_length - 1], [0, side_length - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst_points)
    warped = cv2.warpPerspective(img, M, (side_length, side_length))
    
    # === ステップ2: 格子線の検出（ここからが新しい処理） ===
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_edges = cv2.Canny(warped_gray, 100, 200)

    # Hough変換で直線を検出
    lines = cv2.HoughLinesP(warped_edges, 1, np.pi / 180, threshold=200, minLineLength=100, maxLineGap=20)
    if lines is None:
        print("格子線を検出できませんでした。単純な9等分を試みます。")
        # （フォールバック処理として前回のコードをここに記述することも可能）
        return

    # 垂直線と水平線を分ける
    vertical_x = []
    horizontal_y = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # 垂直に近い線
        if abs(angle) > 85 and abs(angle) < 95:
            vertical_x.append(x1)
            cv2.line(warped, (x1, y1), (x2, y2), (0, 0, 255), 1) # 赤色で描画
        # 水平に近い線
        elif abs(angle) < 5:
            horizontal_y.append(y1)
            cv2.line(warped, (x1, y1), (x2, y2), (255, 0, 0), 1) # 青色で描画

    # === ステップ3: 直線の整理と絞り込み ===
    v_lines = _cluster_lines(vertical_x, axis=0)
    h_lines = _cluster_lines(horizontal_y, axis=1)
    
    # 盤の境界線を含め、10本の線が検出できているか確認
    if len(v_lines) < 10 or len(h_lines) < 10:
        print(f"十分な格子線が見つかりませんでした。(垂直: {len(v_lines)}本, 水平: {len(h_lines)}本)")
        # 確認用に検出した線が描画された画像を表示
        cv2.imshow("Detected Lines", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # === ステップ4: 座標の特定と分割 ===
    for r in range(9):
        for c in range(9):
            y1 = h_lines[r]
            y2 = h_lines[r+1]
            x1 = v_lines[c]
            x2 = v_lines[c+1]
            
            cell_img = warped[y1:y2, x1:x2]
            
            if cell_img.size > 0:
                filename = os.path.join(output_dir, f'cell_{r}_{c}.png')
                cv2.imwrite(filename, cell_img)
            
    print(f"格子線を基に81マスを '{output_dir}' に保存しました。")
    
    # 確認用に検出した線が描画された画像を表示
    cv2.imshow("Detected Lines", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    input_image_file = 'shogi_board_sample.jpg' # 入力画像ファイルを指定
    detect_shogi_grid_and_crop(input_image_file)