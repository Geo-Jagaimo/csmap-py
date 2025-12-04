import numpy as np


def rgbify(arr: np.ndarray, method, scale: tuple[float, float] = None) -> np.ndarray:
    """ndarrayをRGBに変換する
    - arrは変更しない
    - ndarrayのshapeは、(4, height, width) 4はRGBA
    """

    _min = arr.min() if scale is None else scale[0]
    _max = arr.max() if scale is None else scale[1]

    # -x ~ x を 0 ~ 1 に正規化
    arr = (arr - _min) / (_max - _min)
    # clamp
    arr = np.where(arr < 0, 0, arr)
    arr = np.where(arr > 1, 1, arr)

    # 3次元に変換
    rgb = method(arr)
    return rgb


def slope_red(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = 247 - arr * 113  # R: 247 -> 134
    rgb[1, :, :] = 213 - arr * 185  # G: 213 -> 28
    rgb[2, :, :] = 213 - arr * 180  # B: 213 -> 33
    rgb[3, :, :] = 255
    return rgb


def slope_blackwhite(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = 246 - arr * 210  # R: 246 -> 36
    rgb[1, :, :] = 246 - arr * 210  # G: 246 -> 36
    rgb[2, :, :] = 246 - arr * 210  # B: 246 -> 36
    rgb[3, :, :] = 255  # A
    return rgb


def curvature_blue(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = 42 + arr * 166  # R: 42 -> 208
    rgb[1, :, :] = 95 + arr * 128  # G: 95 -> 223
    rgb[2, :, :] = 131 + arr * 99  # B: 131 -> 230
    rgb[3, :, :] = 255
    return rgb


def curvature_redyellowblue(arr: np.ndarray) -> np.ndarray:
    # value:0-1 to: blue -> yellow -> red
    # interpolate between blue and yellow, and yellow and red, by linear

    # 0-0.5: blue -> yellow
    rgb1 = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb1[0, :, :] = 50 + arr * 205 * 2  # R: 50 -> 255
    rgb1[1, :, :] = 96 + arr * 158 * 2  # G: 96 -> 254
    rgb1[2, :, :] = 207 - arr * 17 * 2  # B: 207 -> 190

    # 0.5-1: yellow -> red
    rgb2 = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb2[0, :, :] = 255 - (arr * 2 - 1) * 57  # R: 255 -> 198
    rgb2[1, :, :] = 254 - (arr * 2 - 1) * 182  # G: 254 -> 72
    rgb2[2, :, :] = 190 - (arr * 2 - 1) * 131  # B: 190 -> 59

    # blend
    rgb = np.where(arr < 0.5, rgb1, rgb2)
    rgb[3, :, :] = 255

    return rgb


def height_blackwhite(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = 36 + arr * 210  # R: 36 -> 246
    rgb[1, :, :] = 36 + arr * 210  # G: 36 -> 246
    rgb[2, :, :] = 36 + arr * 210  # B: 36 -> 246
    rgb[3, :, :] = 255
    return rgb


def blend(
    dem_bw: np.ndarray,
    slope_red: np.ndarray,
    slope_bw: np.ndarray,
    curvature_blue: np.ndarray,
    curvature_ryb: np.ndarray,
    blend_params: dict = {
        "slope_bw": 0.25,  # 傾斜（白黒）
        "curvature_ryb": 0.25,  # 曲率（青黄赤）
        "slope_red": 0.25,  # 傾斜（白茶）
        "curvature_blue": 0.125,  # 曲率（紺白）
        "dem": 0.125,  # 標高（白黒）
    },
    nodata_mask: np.ndarray = None,
) -> np.ndarray:
    """blend all rgb
    全てのndarrayは同じshapeであること
    DEMを用いて処理した他の要素は、DEMよりも1px内側にpaddingされているので
    あらかじめDEMのpaddingを除外しておく必要がある
    """
    _blend = np.zeros((4, dem_bw.shape[0], dem_bw.shape[1]), dtype=np.uint8)
    _blend = (
        dem_bw * blend_params["dem"]
        + slope_red * blend_params["slope_red"]
        + slope_bw * blend_params["slope_bw"]
        + curvature_blue * blend_params["curvature_blue"]
        + curvature_ryb * blend_params["curvature_ryb"]
    )
    _blend = _blend.astype(np.uint8)  # force uint8

    # 色の鮮明化: 各チャンネルを0-255の範囲にストレッチ
    for i in range(3):  # R, G, B
        channel_min = _blend[i, :, :].min()
        channel_max = _blend[i, :, :].max()
        if channel_max > channel_min:
            _blend[i, :, :] = (
                (_blend[i, :, :] - channel_min) / (channel_max - channel_min) * 255
            ).astype(np.uint8)

    # alpha channel: Nodata areas are must be transparent (Alpha=0)
    if nodata_mask is not None:
        _blend[3, :, :] = np.where(nodata_mask, 0, 255)
    else:
        _blend[3, :, :] = 255

    return _blend
