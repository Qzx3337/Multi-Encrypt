import os
import torch
import torch.nn as nn
import torch.onnx


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 定义 ActorCriticPolicy 模型
class ActorCriticPolicy(nn.Module):
    def __init__(self):
        super(ActorCriticPolicy, self).__init__()

        # FlattenExtractor (共享特征提取器)
        self.features_extractor = nn.Flatten(start_dim=1, end_dim=-1)

        # Policy MLP 网络
        self.policy_net = nn.Sequential(
            nn.Linear(in_features=8, out_features=64, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.Tanh()
        )

        # Value MLP 网络
        self.value_net = nn.Sequential(
            nn.Linear(in_features=8, out_features=64, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.Tanh()
        )

        # Action 输出层
        self.action_net = nn.Linear(in_features=64, out_features=3, bias=True)

        # Value 输出层
        self.value_net_output = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, x):
        # 特征提取
        x = self.features_extractor(x)

        # Policy 网络前向传播
        policy_out = self.policy_net(x)
        action_out = self.action_net(policy_out)

        # Value 网络前向传播
        value_out = self.value_net(x)
        value_out = self.value_net_output(value_out)

        return action_out, value_out


# 创建模型实例
model = ActorCriticPolicy()

# 打印模型结构
print(model)

# 导出为 ONNX 格式
dummy_input = torch.randn(1, 8)  # 输入张量 (batch_size=1, input_features=8)
onnx_path = "actor_critic_policy.onnx"

# 导出模型
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["action_output", "value_output"],
    dynamic_axes={"input": {0: "batch_size"},
                  "action_output": {0: "batch_size"},
                  "value_output": {0: "batch_size"}}
)

print(f"Model exported to {onnx_path}")