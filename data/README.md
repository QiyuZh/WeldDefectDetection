# 数据目录约定

```text
data/
├─ raw/
│  ├─ images/      # 原始图片
│  └─ labels/      # LabelImg 导出的 YOLO txt
└─ dataset/
   ├─ images/
   │  ├─ train/
   │  └─ val/
   └─ labels/
      ├─ train/
      └─ val/
```

正常样本无需标注文件；如果存在空标签文件，也会被兼容处理。

