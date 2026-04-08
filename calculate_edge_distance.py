import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的地理距离（单位：千米）"""
    R = 6371  # 地球平均半径（千米）
    
    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # 计算经纬度差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine公式
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def main():
    # 读取边的数据
    edges = pd.read_csv(r"data/edge_info.csv")
    
    # 读取节点坐标数据
    nodes = pd.read_csv(r"data/points_info.csv")
    
    # 获取唯一的节点ID和对应的最新坐标
    nodes = nodes.drop_duplicates(subset='FID', keep='last')[['FID', 'Lon', 'Lat']]
    
    # 计算每条边的距离
    weights = []
    for _, edge in edges.iterrows():
        # 获取起点和终点的坐标
        source_coords = nodes[nodes['FID'] == edge['source']].iloc[0]
        target_coords = nodes[nodes['FID'] == edge['target']].iloc[0]
        
        # 计算距离
        distance = haversine_distance(
            source_coords['Lat'], source_coords['Lon'],
            target_coords['Lat'], target_coords['Lon']
        )
        weights.append(1 / distance)
    
    # 将距离添加到边数据中
    edges['weight'] = weights
    
    # 保存更新后的边数据
    edges.to_csv(r'data/edge_weight.csv', index=False)
    print(f'已计算完成 {len(weights)} 条边的距离并更新到edge.csv文件中')

if __name__ == '__main__':
    main()