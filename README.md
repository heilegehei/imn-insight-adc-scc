# IMN Insight · 腺癌 vs 鳞癌在线工具

在线体验：[IMN Insight](https://imn-insight-adc-scc.streamlit.app/)。

工具采用黑白分区、大标题、蓝色交互和响应式布局。主视觉由炎症细胞、抗体盾牌、营养叶片及分子元素组成，对应 I / M / N 三个层级。界面为英文。

## 功能

- 单例输入、三层 IMN 概率、各层固定阈值对应类别、六项复合指标、输入核对、结果 CSV 下载。
- CSV 批量输入，最多 2,000 行 / 5 MB；保留原始行序，支持 CHOL 等字段别名。
- 空值调用原有确定性插补器；非法字符、无穷值、负值、无效分母等会明确报错，不悄悄生成结果。
- 可一键加载合成演示值。部署包不包含原项目患者 CSV 或已知患者测试样本。

本工具仅适用于已明确为腺癌或鳞癌病例的条件二分类，不用于肺癌筛查、临床诊断或治疗决策。

## 本地启动

需要 Python 3.13。Windows PowerShell 中进入本目录后运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
```

首次运行建立本目录专用 `.venv` 并安装依赖。默认打开地址为 `http://localhost:8502`，启动后在浏览器输入该地址。可加 `-Port 8503` 修改端口，或 `-Install` 重新核验安装依赖。

已有匹配环境时可直接运行：

```powershell
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

macOS / Linux：安装 Python 3.13 后执行 `bash run_mac.sh`。macOS 如 LightGBM 提示缺少 OpenMP，先安装 `libomp`。Windows 如二进制库缺少运行时，安装 Microsoft Visual C++ Redistributable。

## 部署至公网

这是可独立部署的目录，不依赖电脑上的原始数据或上级路径。不要上传整个研究文件夹。

1. 将本目录作为独立 Git 仓库上传至自己的 GitHub。保留 `assets/` 中的模型与 `runtime/`，不要加入患者样本、日志或 `.streamlit/secrets.toml`。
2. 在 Streamlit Community Cloud 选择该仓库与分支，入口填写 `app.py`。
3. 高级设置选择 Python **3.13**。根目录 `requirements.txt` / `packages.txt` 分别安装 Python 依赖和 Linux OpenMP 库。
4. 部署后用页面中的合成例子测试单例和批量预测。公开网址不得接收可识别患者资料。

如将本目录保留在仓库的子路径下，入口应填写实际子路径到 `app.py`，并按托管平台规则配置依赖文件位置。

官方部署说明：[Streamlit Community Cloud 部署指南](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)。

## 模型与单位

主层：I + M + N；主模型：Rotation Forest + Extra Trees → L2 Logistic Stacking。

| 层 | 输入模型变量数 | 固定分类阈值 |
| --- | ---: | ---: |
| I | 4 | 0.7005904515139433 |
| I + M | 6 | 0.7349164121493799 |
| I + M + N | 13 | 0.6194277055171358 |

阈值沿用既有开发集固定阈值，不在应用中重新估计。概率 ≥ 阈值判为腺癌，否则为鳞癌。腺癌 = 1，鳞癌 = 0。

完整输入 27 项：WBC、NEUT、NEUT_PCT、MONO、MONO_PCT、PLT、RDW_CV、CRP、FIB、LDH、LYMPH、LYMPH_PCT、EOS、EOS_PCT、BASO、BASO_PCT、HGB、ALB、TP、GLB、PA、GLU、TG、TC、HDL_C、LDL_C、UA。界面及字典均列明单位。

- 细胞绝对计数：10⁹/L；百分比按 0–100 输入，不输入 0–1 比例。
- CRP：mg/L，`<x` 按 `x/√2`、`>x` 按 `x`；FIB：g/L；LDH：U/L。
- HGB、ALB、TP、GLB：g/L；**PA：mg/dL**。
- GLU、TG、TC、HDL_C、LDL_C：mmol/L；UA：µmol/L。
- TC 可使用 CSV 别名 CHOL，不作数值换算。RDW 为 RDW-CV，不接受 RDW-SD 替代。
- NLR = NEUT/LYMPH；SIRI = NEUT×MONO/LYMPH；LMR = LYMPH/MONO。
- PNI = ALB+5×LYMPH；AGR = ALB/(TP−ALB)；GAR = GLU/ALB。

27 项供原插补路径使用，不表示全部进入最终分类器。最终主层是 7 项直接指标加 6 项复合指标。年龄、性别、吸烟史及肿瘤标志物不进入模型。

## 文件与复现

- `app.py` / `style.css`：交互界面与视觉样式。
- `model_runtime.py`：单位说明、输入检查、原推理路径适配及下载。
- `runtime/frozen_runtime.py` / `runtime/runtime_support/`：从现有模型包原样复制，SHA-256 可核对。
- `assets/top2_stacking_model_package.pkl`：从现有模型制作的仅含推理必需对象的发布副本；不重新训练、不包含患者表格、训练审计或本机路径。
- `assets/model_manifest.json`：模型指纹、版本与阈值来源。
- `tests/`：预测一致性、批量、缺失值、安全输入、界面回归测试。

运行测试：

```powershell
python -m unittest discover -s tests -v
```

若位于原项目中，测试额外读取上级目录已知样本并核对预期结果；独立部署包中没有这些患者样本，此项自动跳过，其余测试正常执行。

工具不会拟合插补器、重选特征、调参、重新校准、优化阈值或改写既有结果。模型文件通过 SHA-256 核验后才加载，不接受上传 pickle 文件。

## 数据处理边界

输入会送达运行应用的服务器，不是纯浏览器端计算；数值与预测仅保存在当前会话内存中，不写入数据文件，不使用共享预测缓存。只有模型对象在进程内复用。下载内容由用户自行保存。清除表单会移除该单例结果；更换批量文件会清除旧批量结果。

不要上传姓名、证件号等身份信息。公开托管时需遵守适用的数据保护要求；本模型不具有医学器械或临床诊断资质。
