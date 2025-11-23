import math

def calculate_angle(a, b, c):
    """
    计算三点之间的夹角（适合膝盖弯曲角度）
    a, b, c 为三维坐标点 (x, y, z)
    返回角度值（单位：度）
    """
    ab = [a[i] - b[i] for i in range(3)]
    cb = [c[i] - b[i] for i in range(3)]
    dot = sum(ab[i] * cb[i] for i in range(3))
    norm_ab = math.sqrt(sum(x ** 2 for x in ab))
    norm_cb = math.sqrt(sum(x ** 2 for x in cb))
    cosine_angle = dot / (norm_ab * norm_cb)
    angle = math.degrees(math.acos(cosine_angle))
    return round(angle, 2)
