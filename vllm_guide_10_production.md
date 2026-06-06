# vLLM 高级指南 10：生产环境部署与监控

## 10.1 Docker 部署

### 基础 Docker 配置

```dockerfile
# Dockerfile
FROM vllm/vllm-openai:latest

WORKDIR /app
COPY model/ ./model/

# 环境变量配置
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV TORCH_CPP_ARCH="9.0"
ENV NCCL_DEBUG=INFO

# 启动命令
CMD ["vllm", "serve", "/app/model", "--host", "0.0.0.0", "--port", "8000"]
```

### 构建和运行

```bash
# 构建镜像
docker build -t my-vllm-service .

# 运行容器
docker run --gpus all \
    --shm-size 32g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    my-vllm-service
```

### Docker 高级配置

```dockerfile
# Dockerfile
FROM vllm/vllm-openai:latest

WORKDIR /app

# 安装额外依赖
RUN pip install prometheus-client openai

# 复制模型（如果本地模型）
COPY ./model /app/model

# 环境变量
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV TORCH_CPP_ARCH="9.0"
ENV NCCL_IB_DISABLE=0
ENV NCCL_NET_GDR_LEVEL=IB

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["vllm", "serve", "/app/model", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--gpu-memory-utilization", "0.9", \
     "--max-model-len", "32768"]
```

### Docker Compose 配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  vllm:
    build: .
    image: my-vllm-service:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    shm_size: '32gb'
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ./config.json:/app/config.json:ro
    environment:
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - CUDA_VISIBLE_DEVICES=0,1,2,3
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 10.2 Kubernetes 部署

### 基本 Deployment 配置

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
  labels:
    app: vllm
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args: 
          - "vllm"
          - "serve"
          - "Qwen/Qwen2.5-7B-Instruct"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        env:
        - name: VLLM_WORKER_MULTIPROC_METHOD
          value: "spawn"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
```

### Service 配置

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
```

### ConfigMap 配置

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-config
data:
  vllm.yaml: |
    model: Qwen/Qwen2.5-7B-Instruct
    tensor-parallel-size: 1
    gpu-memory-utilization: 0.9
    max-model-len: 32768
    enable-chunked-prefill: true
    enable-prefix-caching: true
```

### 多 GPU Deployment 配置

```yaml
# deployment-multi-gpu.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment-tp4
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-tp4
  template:
    metadata:
      labels:
        app: vllm-tp4
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - "vllm"
          - "serve"
          - "meta-llama/Llama-3.1-70B-Instruct"
          - "--tensor-parallel-size"
          - "4"
          - "--gpu-memory-utilization"
          - "0.9"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 4
            memory: "128Gi"
          requests:
            nvidia.com/gpu: 4
            memory: "64Gi"
        env:
        - name: VLLM_WORKER_MULTIPROC_METHOD
          value: "spawn"
        - name: NCCL_DEBUG
          value: "INFO"
```

## 10.3 性能监控

### 启用 Prometheus 指标

```bash
# 启用 Prometheus 指标端点
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-metrics \
    --metric-method=prometheus
```

### OpenTelemetry 集成

```bash
# 使用 OpenTelemetry 导出指标
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-metrics \
    --otlp-columns="prometheus"
```

### 指标端点

```bash
# 获取所有指标
curl http://localhost:8000/metrics

# Prometheus 格式示例
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 5.0

# HELP vllm:num_requests_waiting Number of waiting requests  
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 12.0

# HELP vllm:gpu_cache_usage_perc GPU cache usage percentage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.85
```

## 10.4 关键监控指标

### 核心指标说明

| 指标 | 类型 | 说明 | 正常范围 | 告警阈值 |
|------|------|------|---------|---------|
| `vllm:num_requests_running` | Gauge | 正在处理的请求数 | < max_seqs | > max_seqs |
| `vllm:num_requests_waiting` | Gauge | 等待中的请求数 | < 50 | > 100 |
| `vllm:gpu_cache_usage_perc` | Gauge | GPU KV Cache 使用率 | < 90% | > 95% |
| `vllm:prompt_tokens_total` | Counter | 累计 prompt token 数 | - | - |
| `vllm:generation_tokens_total` | Counter | 累计生成 token 数 | - | - |
| `vllm:time_to_first_token_seconds` | Histogram | TTFT 平均延迟 | < 0.5s | > 1s |
| `vllm:time_per_output_token_seconds` | Histogram | ITL 平均延迟 | < 0.05s | > 0.1s |

### 延迟指标详解

```
TTFT (Time To First Token) - 首 token 延迟
├── 从请求发送到第一个 token 生成的时间
├── 受 prefill 阶段影响
└── 过长说明：prompt 太长或 GPU 负载过高

ITL (Inter-Token Latency) - token 间延迟  
├── 相邻两个 token 的生成时间间隔
├── 主要受 decode 阶段影响
└── 过长说明：batch size 过大或 GPU 资源不足
```

### 自定义监控面板

```python
# Prometheus 配置 (prometheus.yml)
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
```

```yaml
# Grafana Dashboard JSON (片段)
{
  "panels": [
    {
      "title": "Request Throughput",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(vllm:prompt_tokens_total[5m])",
          "legendFormat": "Prompt Tokens/s"
        },
        {
          "expr": "rate(vllm:generation_tokens_total[5m])",
          "legendFormat": "Generation Tokens/s"
        }
      ]
    },
    {
      "title": "GPU Cache Usage",
      "type": "graph",
      "targets": [
        {
          "expr": "vllm:gpu_cache_usage_perc",
          "legendFormat": "Cache Usage %"
        }
      ]
    }
  ]
}
```

## 10.5 健康检查

### API 健康检查

```bash
# 检查服务健康
curl http://localhost:8000/health

# 返回示例
{"status":"healthy","model":"Qwen/Qwen2.5-7B-Instruct"}

# 检查模型列表
curl http://localhost:8000/v1/models

# 返回示例
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-7B-Instruct",
      "object": "model",
      "created": 1234567890,
      "owned_by": "system"
    }
  ]
}
```

### Kubernetes 健康检查配置

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

## 10.6 负载均衡

### Nginx 配置

```nginx
# nginx.conf
upstream vllm_backend {
    least_conn;  # 最少连接优先
    
    server 10.0.0.1:8000 weight=5;
    server 10.0.0.2:8000 weight=5;
    server 10.0.0.3:8000 weight=5;
}

server {
    listen 80;
    server_name api.example.com;

    client_max_body_size 10M;
    
    location /v1/chat/completions {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_read_timeout 300s;
        proxy_connect_timeout 30s;
    }

    location /v1/completions {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

### HAProxy 配置

```
# haproxy.cfg
global
    log stdout format raw local0

defaults
    mode http
    timeout connect 10s
    timeout client 300s
    timeout server 300s

frontend vllm_front
    bind *:80
    default_backend vllm_back

backend vllm_back
    balance leastconn
    server vllm1 10.0.0.1:8000 check inter 10s fall 2 rise 3
    server vllm2 10.0.0.2:8000 check inter 10s fall 2 rise 3
    server vllm3 10.0.0.3:8000 check inter 10s fall 2 rise 3
```

## 10.7 自动扩缩容

### Kubernetes HPA 配置

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
```

### 基于自定义指标的 HPA

```yaml
# custom-metrics-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa-custom
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: vllm_requests_waiting
      target:
        type: AverageValue
        averageValue: "50"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
```

### 弹性伸缩脚本

```python
#!/usr/bin/env python3
# autoscale.py
import requests
import time
import subprocess

VLLM_API = "http://10.0.0.1:8000"
MAX_REPLICAS = 10
MIN_REPLICAS = 2
TARGET_WAITING = 50

def get_metrics():
    """获取当前指标"""
    try:
        resp = requests.get(f"{VLLM_API}/metrics")
        metrics = resp.text
        # 解析 waiting 请求数
        for line in metrics.split('\n'):
            if 'vllm:num_requests_waiting' in line:
                return float(line.split()[-1])
    except:
        return 0
    return 0

def get_current_replicas():
    """获取当前副本数"""
    result = subprocess.run(
        ["kubectl", "get", "deployment", "vllm-deployment", 
         "-o", "jsonpath={.spec.replicas}"],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

def scale(deployment, replicas):
    """伸缩副本数"""
    subprocess.run(
        ["kubectl", "scale", "deployment", deployment, 
         f"--replicas={replicas}"],
        check=True
    )

def main():
    while True:
        waiting = get_metrics()
        current = get_current_replicas()
        
        if waiting > TARGET_WAITING * 1.2 and current < MAX_REPLICAS:
            new_replicas = min(current + 1, MAX_REPLICAS)
            print(f"Scaling up: {current} -> {new_replicas}")
            scale("vllm-deployment", new_replicas)
        elif waiting < TARGET_WAITING * 0.8 and current > MIN_REPLICAS:
            new_replicas = max(current - 1, MIN_REPLICAS)
            print(f"Scaling down: {current} -> {new_replicas}")
            scale("vllm-deployment", new_replicas)
        
        time.sleep(30)

if __name__ == "__main__":
    main()
```

## 10.8 安全最佳实践

### API 认证

```bash
# 启用 API Key 认证
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --api-key "your-secure-api-key-here" \
    --allowed-origins "https://your-app.com" \
    --allowed-methods "POST" \
    --allowed-headers "*"
```

### 请求来源限制

```yaml
# Nginx IP 白名单
server {
    location /v1/chat/completions {
        allow 10.0.0.0/8;
        allow 192.168.0.0/16;
        deny all;
        
        proxy_pass http://vllm_backend;
    }
}
```

### 速率限制

```nginx
# Nginx 速率限制
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

server {
    location /v1/chat/completions {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://vllm_backend;
    }
}
```

### TLS 配置

```yaml
# Kubernetes TLS Secret
apiVersion: v1
kind: Secret
metadata:
  name: vllm-tls
type: kubernetes.io/tls
data:
  tls.crt: <base64-encoded-cert>
  tls.key: <base64-encoded-key>
---
# Ingress 配置
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: vllm-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-service
            port:
              number: 8000
```

## 10.9 备份与恢复

### 模型缓存管理

```bash
# 查看模型缓存位置
ls ~/.cache/huggingface/hub/

# 模型缓存结构
# ~/.cache/huggingface/hub/
# ├── models--Qwen--Qwen2.5-7B-Instruct/
# │   ├── blobs/
# │   ├── refs/
# │   └── snapshots/

# 清理未使用的模型缓存
pip install huggingface_hub
huggingface-cli delete-cache
```

### 模型备份

```bash
# 备份模型到本地
cp -r ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct /backup/

# 使用 rsync 增量备份
rsync -avz ~/.cache/huggingface/hub/ /backup/huggingface/

# 备份到对象存储
aws s3 sync ~/.cache/huggingface/hub/ s3://my-bucket/huggingface/
```

### ModelScope 缓存

```bash
# 使用 ModelScope 替代 HuggingFace
export VLLM_USE_MODELSCOPE=True

# 或在代码中
from vllm import LLM

llm = LLM(
    model="qwen/Qwen2.5-7B-Instruct",
    model_scope="qwen",  # 使用 ModelScope
)
```

### 恢复模型

```bash
# 从本地备份恢复
cp -r /backup/models--Qwen--Qwen2.5-7B-Instruct ~/.cache/huggingface/hub/

# 从对象存储恢复
aws s3 sync s3://my-bucket/huggingface/ ~/.cache/huggingface/hub/
```

## 10.10 故障排查

### 常见问题与解决方案

| 问题 | 排查步骤 | 解决方案 |
|------|---------|---------|
| 服务启动失败 | 1. 检查 GPU 可用性<br>2. 检查 CUDA 版本<br>3. 检查模型下载 | 确保 GPU 可见，更新驱动，预下载模型 |
| OOM | 1. 检查 gpu_memory_utilization<br>2. 检查 max_model_len | 降低 gpu_memory_utilization，减小 max_model_len |
| 高延迟 | 1. 检查 GPU 利用率<br>2. 检查 batch size<br>3. 查看 TTFT/ITL | 增加 GPU，考虑启用推测解码 |
| 请求超时 | 1. 检查网络连通性<br>2. 查看 vLLM 日志 | 增加 timeout 设置，检查防火墙 |
| 缓存命中率低 | 1. 检查 prefix caching 配置<br>2. 查看请求模式 | 固定系统提示，启用缓存 |

### 排查命令

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 vLLM 日志
docker logs <container_id>

# 查看 Kubernetes pod 状态
kubectl get pods -l app=vllm
kubectl describe pod <pod_name>
kubectl logs <pod_name>

# 检查网络连接
curl -v http://localhost:8000/health
```

### 日志分析

```bash
# 启用详细日志
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --enable-metrics \
    -log_level=DEBUG

# 查看最近日志
kubectl logs -f <pod_name> --tail=100

# 分析 Prometheus 指标趋势
curl http://localhost:8000/metrics | grep vllm:num_requests
```

### 性能诊断

```python
# 性能诊断代码示例
from vllm import LLM, SamplingParams
import time

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 测量延迟
sampling_params = SamplingParams(max_tokens=100)

start = time.time()
outputs = llm.generate(["Hello, world!"], sampling_params)
end = time.time()

print(f"Total time: {end - start:.3f}s")
print(f"Output: {outputs[0].outputs[0].text}")

# 查看详细采样信息
print(f"Prompt tokens: {outputs[0].prompt_token_ids}")
print(f"Output tokens: {outputs[0].outputs[0].token_ids}")
print(f"Stop reason: {outputs[0].outputs[0].stop_reason}")
```

## 10.11 灾难恢复

### 备份策略

```yaml
# CronJob 定期备份
apiVersion: batch/v1
kind: CronJob
metadata:
  name: vllm-backup
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: amazon/aws-cli
            command:
            - sh
            - -c
            - |
              aws s3 sync /root/.cache/huggingface/hub/ s3://my-bucket/vllm-models/
          restartPolicy: OnFailure
```

### 故障切换

```yaml
# 多区域部署
---
# primary-region/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-primary
spec:
  replicas: 3
  # ...

---
# secondary-region/deployment.yaml  
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-secondary
spec:
  replicas: 1
  # ...
```

### 恢复流程

```bash
#!/bin/bash
# restore.sh

BACKUP_SOURCE="s3://my-bucket/vllm-models/"
LOCAL_CACHE="/root/.cache/huggingface/hub/"

echo "Starting recovery from backup..."
aws s3 sync $BACKUP_SOURCE $LOCAL_CACHE

echo "Verifying model integrity..."
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print('Model verified successfully')
"

echo "Restarting vLLM service..."
systemctl restart vllm

echo "Checking service health..."
curl -f http://localhost:8000/health && echo "Service restored!"
```

## 10.12 容量规划

### 容量估算公式

```
单 GPU 容量 = (GPU 显存 - 模型参数显存) / KV Cache 单请求显存

支持的并发请求数 = 容量 / 单请求 KV Cache

示例:
- A100 80GB
- Llama-7B FP16: 14GB
- 可用显存: 80 - 14 = 66GB
- 单请求 (4096 tokens): ~2GB
- 并发容量: 66 / 2 = 33 请求
```

### 扩展规划表

| 日均请求量 | 峰值 QPS | 推荐 GPU | 副本数 | 配置 |
|-----------|---------|---------|--------|------|
| 10万 | 10 | A10G | 1 | 7B |
| 50万 | 50 | A100 40GB | 2 | 7B |
| 100万 | 100 | A100 80GB | 2 | 13B |
| 500万 | 500 | A100 80GB x4 | 8 | 70B |

### 成本优化建议

1. **使用量化模型**：INT8/INT4 量化减少显存占用
2. **启用前缀缓存**：减少重复计算
3. **合理设置 max_tokens**：避免预留过多
4. **使用 Spot 实例**：成本降低 50-70%
5. **自动扩缩容**：根据负载动态调整

```yaml
# Spot 实例配置示例
spec:
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/lifecycle: spot
      tolerations:
      - key: "node.kubernetes.io/lifecycle"
        operator: "Equal"
        value: "spot"
```
