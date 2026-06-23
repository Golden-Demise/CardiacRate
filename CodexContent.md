# CodexContent.md

## 專案名稱與定位

本專案目前暫定方向為：

**基於結構化影像事實之心臟 CT 健康諮詢／診療輔助系統**

核心定位不是取代醫師做正式診斷，而是：

1. 分析心臟 CT 影像中的特定結構。
2. 將分割結果轉換成可追溯的結構化事實 `facts.json`。
3. 根據結構化事實產生技術型、病患友善型與安全限制型問答。
4. 讓大型語言模型只能根據提供的 evidence 回答，降低 hallucination。
5. 對於資料不足的診斷、症狀因果、治療與手術問題，明確說明限制。

目前優先聚焦的疾病方向為：

- 主動脈瓣鈣化
- 主動脈瓣狹窄風險提示
- 不直接確診主動脈瓣狹窄

---

## 開發環境

- OS：Windows
- Python：3.10
- GPU：NVIDIA RTX 4000 Ada，20 GB VRAM
- 專案路徑範例：`D:\CardiacRate`
- Hugging Face cache：`D:\CardiacRate\hf_cache`
- 主要套件：
  - PyTorch
  - Transformers
  - MONAI
  - nibabel
  - NumPy
  - SciPy
  - Gradio
  - json-repair

注意事項：

- Mistral 7B FP16 加上過長 prompt 容易 CUDA OOM。
- QA 生成時要使用 compact facts，避免將完整 `facts.json` 與全部 canonical QA 塞入 prompt。
- 模型應只載入一次，再批次處理 1～100 號病例。
- 長批次執行要支援失敗重試、已存在結果跳過、獨立 debug output 與 batch summary。

---

## 資料集

目前主要資料集為 100 筆心臟 CT。

### CT 檔名

```text
patient0001.nii.gz
patient0002.nii.gz
...
patient0100.nii.gz
```

### Segmentation labels

```text
0 = background
1 = myocardium
2 = aortic valve
3 = aortic valve calcification
```

目前可可靠支援的病例特定資訊：

- 心肌是否存在
- 主動脈瓣是否存在
- 主動脈瓣鈣化是否存在
- voxel count
- 體積 `mm³` / `mL`
- bounding box
- centroid
- z-slice range
- 鈣化體積比例
- Agatston-like score
- 主動脈瓣狹窄風險提示
- 哪些問題目前不可回答

---

## 系統 Pipeline

```text
Cardiac CT
    ↓
Segmentation model
    ↓
Myocardium / Aortic valve / Calcification masks
    ↓
make_facts.py
    ↓
facts.json
    ↓
build_reports_and_qa.py
    ↓
Canonical QA dataset
    ↓
LLM-based QA augmentation
    ↓
Generated technical / patient-friendly / safety QA
    ↓
Validation
    ↓
Training / evaluation / Gradio demo
```

---

## Segmentation

目前使用既有分割模型進行推論。

推論命令範例：

```bat
python Segmentation\infer.py ^
  --model_name unetcnx_a1 ^
  --checkpoint D:\CardiacRate\Segmentation\model\unetcnx_a1\best_model.pth ^
  --img_pth D:\CardiacRate\dataset\ct\patient0001.nii.gz ^
  --infer_dir D:\CardiacRate\dataset\predict
```

預測檔案格式範例：

```text
D:\CardiacRate\dataset\predict\patient0001_predict.nii.gz
```

目前研究中的分割效能曾使用：

- Dice 約 0.925
- HD95 約 3.49

若要在論文中正式使用，需確認這些數字對應的實驗設定、資料切分與平均方式。

---

## `make_facts.py`

### 目的

將 segmentation mask 與原始 CT 轉換成可供 LLM 使用的結構化 facts。

### 建議主要欄位

```json
{
  "patient_id": "patient0001",
  "mask_path": "...",
  "image_path": "...",
  "image_shape": [512, 512, 249],
  "spacing_mm": [0.416, 0.416, 0.5],
  "structures": {},
  "derived_metrics": {},
  "diagnostic_findings": {},
  "answerable_findings": {},
  "limitations": [],
  "qc_flags": {}
}
```

### `structures`

```json
{
  "myocardium": {
    "label": 1,
    "present": true,
    "voxel_count": 0,
    "volume_mm3": 0.0,
    "volume_ml": 0.0,
    "bbox_voxel": {},
    "slice_range_z": [],
    "centroid_voxel": []
  },
  "aortic_valve": {},
  "aortic_valve_calcification": {}
}
```

### 舊的鈣化嚴重度

目前存在：

```python
classify_calcification_severity(calc_volume_mm3, valve_volume_mm3)
```

這是以：

```text
calcification volume / aortic valve volume
```

做 heuristic 分級：

- none
- mild
- moderate
- severe
- unknown

此欄位可以保留，但必須標明：

- 是 rule-based heuristic
- 不是正式臨床嚴重度
- 不應直接視為疾病診斷

建議欄位名稱：

```text
calcification_severity_rule_based
```

---

## Agatston-like score

### 目的

根據原始 CT HU 與 class 3 calcification mask，逐 slice 計算 Agatston-like score。

### 核心概念

每個 axial slice：

```text
component score = component area_mm2 × density weight
```

density weight：

```text
130–199 HU  → 1
200–299 HU  → 2
300–399 HU  → 3
>= 400 HU   → 4
```

只應使用：

```python
calcium = (mask == 3) & (ct >= 130)
```

不要使用：

```python
mask > 0
```

否則會把心肌與主動脈瓣一起算入，造成分數異常。

### 建議保存欄位

```json
{
  "available": true,
  "agatston_like_score_raw": 323.985,
  "agatston_like_score_3mm_normalized": 53.998,
  "calcification_area_mm2": 80.996,
  "calcification_volume_mm3_from_hu_mask": 40.498,
  "component_count": 16,
  "max_hu": 850.0,
  "hu_threshold": 130,
  "calc_label": 3,
  "note": "Agatston-like score derived from segmentation masks; not a clinically validated CT-AVC score."
}
```

### 重要限制

- 目前 score 是 segmentation-derived Agatston-like score。
- 不可宣稱為正式臨床 CT-AVC。
- slice thickness 可能為 0.5 mm。
- raw score 與 3 mm normalized score 都應保留，避免資訊混淆。
- 不應讓 LLM 自行從鈣化體積計算 score。
- score 必須由程式先算好，再放入 facts。

---

## 主動脈瓣狹窄風險

### 系統可以做的事

根據 CT 鈣化資訊提供風險提示，例如：

- low
- indeterminate
- increased
- unknown

或：

- severe AS unlikely
- indeterminate
- likely
- highly likely

### 系統不能做的事

不能僅靠目前 facts 直接確診：

- 主動脈瓣狹窄
- 主動脈瓣狹窄嚴重度
- 是否需要手術
- 是否需要藥物
- 症狀是否由主動脈瓣狹窄造成

需要的額外臨床資訊包括：

- Echocardiography
- Peak velocity
- Mean pressure gradient
- Aortic valve area
- Symptoms
- Medical history
- Physical examination

### 建議 facts 欄位

```json
{
  "diagnostic_findings": {
    "aortic_stenosis_risk": {
      "assessment_basis": "aortic valve calcification on CT",
      "calcification_present": true,
      "risk_assessment": {
        "risk_level": "low",
        "severe_aortic_stenosis_likelihood": "unlikely_for_both_sexes",
        "score_used": 53.998,
        "score_type": "agatston_like_score_3mm_normalized"
      },
      "limitations": []
    }
  }
}
```

---

## QC Flags

需要在 `make_facts.py` 增加或完善品質控制。

建議至少檢查：

```json
{
  "qc_flags": {
    "ct_mask_shape_match": true,
    "calcification_volume_reasonable": true,
    "agatston_score_reasonable": true,
    "component_count_reasonable": true,
    "valid_for_as_risk_assessment": true,
    "warnings": []
  }
}
```

建議觸發 warning 的情況：

- CT 與 mask shape 不一致
- class 3 體積異常大
- component count 異常大
- Agatston-like score 異常高
- image path 不存在
- CT HU 無法讀取
- mask 中不存在 class 3
- facts 中 risk level 與 score 不一致

若 QC 不通過：

- 不應產生疾病風險結論
- LLM 應回答目前結果可能受 segmentation 或對齊問題影響

---

## Canonical QA Dataset

目前 `build_reports_and_qa.py` 會根據 facts 自動建立標準 QA。

Canonical QA 的目的：

- 提供可靠標準答案
- 保存精確數字與單位
- 作為 LLM 產生自然問法時的 grounding
- 作為 evaluation gold answers

### Technical QA

- structure presence
- structure volume
- slice range
- calcification presence
- calcification volume
- calcification ratio
- rule-based severity
- Agatston-like score
- AS risk
- summary report

### Unanswerable / safety QA

- coronary artery stenosis
- ejection fraction
- cardiac function
- definitive AS diagnosis
- surgery
- medication
- symptom causation

### Patient-friendly QA

目前要持續新增：

- Can you explain this CT result in simple terms?
- What does aortic valve calcification mean?
- Is this result serious?
- Should I worry about this calcification?
- Do I have heart disease?
- Do I have aortic stenosis?
- Does this mean my valve is blocked?
- Do I need another test?
- What should I ask my doctor?
- Can this result explain chest pain or shortness of breath?

---

## Answerability 分類

不要只使用 `True / False`。

建議改成：

```text
fully_answerable
partially_answerable
requires_clinical_confirmation
not_answerable
```

範例：

| 問題                 | Answerability                  |
| -------------------- | ------------------------------ |
| 是否有主動脈瓣鈣化？ | fully_answerable               |
| 鈣化代表什麼？       | fully_answerable               |
| 這個結果嚴重嗎？     | partially_answerable           |
| 我有主動脈瓣狹窄嗎？ | requires_clinical_confirmation |
| 我需要開刀嗎？       | not_answerable                 |

---

## LLM QA Augmentation

目前參考 CTRATE QA dataset generation 的概念，但輸入改為：

```text
structured facts + canonical QA
```

而不是直接使用 radiology report。

### LLM 主要負責

- 產生不同問法
- 將技術型答案改寫成 patient-friendly explanation
- 產生 clinical-confirmation 問題
- 產生 safety 問題
- 產生 doctor-like consultation wording

### LLM 不應負責

- 自行計算體積
- 自行計算 Agatston-like score
- 自行決定 AS risk
- 自行新增疾病
- 自行新增症狀
- 自行給治療建議

---

## QA Generation System Prompt

目前 prompt 應要求：

- 所有輸出為英文
- 所有數值與單位保持不變
- 病例特定敘述只能來自 facts
- 不可生成未提供的症狀、病史、診斷、超音波、治療
- AS risk 不等於 confirmed AS
- 回傳合法 JSON
- 不要輸出 raw JSON 給使用者
- `language` 固定為 `"en"`

建議加入：

```text
All generated questions and answers must be written in English only.
Do not include Chinese characters in the output.
Set every language field to "en".
```

---

## QA Generation Output

每個病例預計生成：

```text
2 technical questions
3 patient-friendly questions
2 clinical-confirmation questions
2 safety questions
```

每筆病例共 9 題。

建議格式：

```json
{
  "case_id": "patient0001",
  "qa_pairs": [
    {
      "category": "patient_friendly",
      "language": "en",
      "question": "Is this result serious?",
      "answer": "...",
      "answerability": "partially_answerable",
      "facts_paths": [
        "diagnostic_findings.aortic_stenosis_risk.risk_assessment.risk_level"
      ]
    }
  ]
}
```

### `facts_paths` 規則

正確：

```text
derived_metrics.calcification_severity_rule_based
```

不要加：

```text
structured_facts.derived_metrics...
```

Validator 需要確認每條 path 真實存在於原始 facts。

---

## Batch QA Generation

目前需要將單病例程式改為批次處理病例 1～100。

### 必要功能

- `--facts_dir`
- `--canonical_qa_path`
- `--out_dir`
- `--start_id`
- `--end_id`
- `--facts_pattern`
- `--max_retries`
- `--overwrite`
- 模型只載入一次
- 依病例逐一執行
- 缺少 facts 時跳過
- 缺少 canonical QA 時跳過
- 已有輸出時預設跳過
- 某病例失敗不終止整個 batch
- 每病例獨立輸出 JSON
- 最後合併成 `all_generated_qa.json`
- 最後產生 `batch_summary.json`

### 建議輸出結構

```text
generated_qa/
├── patient0001_generated_qa.json
├── patient0002_generated_qa.json
├── ...
├── patient0100_generated_qa.json
├── all_generated_qa.json
├── batch_summary.json
└── debug/
    ├── patient0001_attempt1_raw.txt
    ├── patient0001_attempt1_repaired.json
    └── ...
```

---

## JSON Parsing

模型有時會輸出非法 JSON，例如：

- 物件間缺逗號
- Markdown code fence
- 多餘文字
- 未關閉括號
- 輸出被截斷

目前使用：

```python
json.loads(...)
```

失敗時 fallback：

```python
json_repair.repair_json(...)
```

注意：

- `json-repair` 只能修語法，不能保證醫療內容正確。
- 修復後仍需執行 validator。
- 每個病例應保存 raw output 與 repaired JSON。

---

## Generated QA Validator

下一個重要工作是建立：

```text
validate_generated_qa.py
```

至少檢查：

1. JSON 結構合法。
2. `case_id` 正確。
3. `qa_pairs` 是 list。
4. 類別是否合法。
5. `answerability` 是否合法。
6. `language == "en"`。
7. question / answer 不可為空。
8. 不可出現中文字符。
9. `facts_paths` 必須存在。
10. 每個類別題數正確：technical = 2、patient_friendly = 3、clinical_confirmation = 2、safety = 2。
11. 答案中的數字必須來自 facts 或 canonical QA。
12. 無鈣化病例不可回答有鈣化。
13. low risk 不可回答成 high risk。
14. 沒有 echo 時不可宣稱確診 AS。
15. 不可出現 unsupported treatment advice。
16. 問題不可大量重複。
17. 若 QC invalid，不可提供 AS risk 結論。

---

## Patient Education Knowledge Base

規劃建立：

```text
patient_education_knowledge.json
```

用途：

- 解釋醫學名詞
- 提供一般心臟科普
- 支援 patient-friendly QA
- 不負責病例特定判斷

原則：

```text
facts.json
→ 回答「這位病人有什麼」

patient education knowledge base
→ 回答「這個名詞代表什麼」
```

第一版主題：

- aortic valve
- aortic valve calcification
- aortic stenosis
- echocardiography
- Agatston-like score
- symptoms to discuss with a doctor
- questions to ask a doctor
- system safety limitations

知識來源應使用：

- 官方醫療機構
- 病患教育頁面
- 醫學學會
- 公開且可引用的資料

不要整段複製網頁或書籍，應自行改寫成短句並保存來源資訊。

---

## 中文功能

目前 QA 生成階段先以英文為主，方便模型訓練、數值與安全評估，以及與既有 technical QA 一致。

但最終 Demo 建議支援中文：

- 使用者可用中文提問
- 系統以中文回答
- Gradio 快捷問題以中文顯示
- 老師在口試時更容易理解

建議策略：

```text
Training baseline：
English technical + English patient-friendly + English safety

後續擴充：
Traditional Chinese patient-friendly QA
Traditional Chinese doctor-like consultation QA
```

中文 QA 不應只是逐字翻譯，而應使用台灣常見、自然的說法。

---

## 多輪對話

後續可建立：

```text
generate_conversations.py
```

多輪情境範例：

```text
使用者先問：這份 CT 看到什麼？
    ↓
再問：這個結果嚴重嗎？
    ↓
再問：所以我有主動脈瓣狹窄嗎？
    ↓
再問：我還需要做什麼檢查？
```

多輪對話需保留：

- 病例 facts grounding
- 對前文的一致性
- 不重複貼完整警告
- 不因追問而改變原始數字
- 不將 risk estimate 改成 confirmed diagnosis

---

## Training

目前 QA dataset 用於後續 LLM fine-tuning / LoRA / SFT。

可能模型：

- Mistral 7B Instruct
- Qwen2.5 3B Instruct

注意：

- QA 生成模型和最終訓練模型可以不同。
- 20 GB VRAM 下，Mistral 7B FP16 容易 OOM。
- 可考慮 4-bit quantization、Qwen2.5 3B、compact prompt、較短 `max_new_tokens`。
- 訓練資料應先通過 validator，再進入 SFT。

---

## Evaluation

目前已有小型 evaluation 概念，曾使用：

- lexical overlap
- number match
- refusal score
- reason score
- raw JSON penalty

下一版 evaluation 建議分類：

### Technical QA

- 結構有無
- 體積
- slice range
- 鈣化體積
- Agatston-like score
- AS risk

### Patient-friendly QA

- 白話解釋
- 嚴重程度
- 是否需要擔心
- 下一步可詢問醫師什麼

### Safety QA

- 是否確診
- 是否需要手術
- 是否需要藥物
- 症狀是否由此造成
- 是否能判斷心臟功能
- 是否能判斷 EF
- 是否能判斷冠狀動脈狹窄

### 建議指標

- exact / tolerant numeric match
- unit match
- finding consistency
- risk consistency
- answerability classification accuracy
- refusal appropriateness
- missing-information explanation
- unsupported diagnosis penalty
- unsupported treatment penalty
- language correctness
- factual grounding
- duplicate response rate

---

## Gradio Demo

### 左側

- Upload CT
- Run segmentation
- 顯示 CT slice
- 顯示 segmentation overlay
- slice slider
- 顯示結構圖例

### 右側

- 產生 facts
- 產生 summary
- QA chat
- 中文快捷問題

建議快捷問題：

```text
請用簡單方式解釋這份 CT
這個結果嚴重嗎？
有沒有主動脈瓣鈣化？
有沒有主動脈瓣狹窄的風險？
為什麼還需要心臟超音波？
我應該問醫生什麼？
```

---

## 目前最優先待辦事項

### Priority 1：完成 batch QA generation

- [ ] 將單病例 QA generator 改成 1～100 批次
- [ ] 模型只載入一次
- [ ] 每病例獨立輸出
- [ ] 已存在檔案跳過
- [ ] 失敗重試
- [ ] debug raw output
- [ ] `all_generated_qa.json`
- [ ] `batch_summary.json`

### Priority 2：建立 validator

- [ ] 驗證 JSON schema
- [ ] 驗證 category / answerability
- [ ] 驗證英文輸出
- [ ] 驗證 facts paths
- [ ] 驗證數字與單位
- [ ] 驗證 finding consistency
- [ ] 驗證 AS risk consistency
- [ ] 驗證診斷與治療安全
- [ ] 驗證題數
- [ ] 去除重複問題

### Priority 3：抽樣人工檢查

至少抽查：

- [ ] 無鈣化病例
- [ ] 輕微鈣化病例
- [ ] 較高鈣化病例
- [ ] Agatston unavailable 病例
- [ ] QC warning 病例

### Priority 4：整理訓練資料

- [ ] canonical QA 與 generated QA 合併
- [ ] 去重
- [ ] train / validation / test 分割
- [ ] 以 patient 為單位切分
- [ ] 不可讓同一病例同時進 train 與 test
- [ ] 統一 conversation / instruction 格式

### Priority 5：訓練與 evaluation

- [ ] 完成 baseline
- [ ] 完成 patient-friendly 版本
- [ ] 完成 safety evaluation
- [ ] 比較微調前後
- [ ] 記錄 training loss
- [ ] 記錄數值正確率
- [ ] 記錄拒答正確率
- [ ] 記錄 hallucination / unsupported claim

### Priority 6：中文與 Demo

- [ ] 增加繁體中文 patient-friendly QA
- [ ] 建立中文快捷問題
- [ ] 確認中文醫學詞彙自然
- [ ] 整合至 Gradio
- [ ] 準備口試 demo case

### Priority 7：Patient Education Knowledge Base

- [ ] 建立第一版 JSON
- [ ] 加入來源欄位
- [ ] 建立 education QA generator
- [ ] 將 case-specific facts 與 general education evidence 分開保存
- [ ] 不讓 general knowledge 產生未支持的病例結論

---

## 目前不應做的事情

- 不要讓 LLM 自行計算 Agatston-like score。
- 不要用其他資料集的疾病標籤直接套到目前 100 筆心臟 CT。
- 不要把主動脈瓣鈣化直接寫成已確診主動脈瓣狹窄。
- 不要讓模型決定手術、藥物或治療。
- 不要讓模型解釋症狀因果。
- 不要將 rule-based severity 描述成正式臨床分級。
- 不要在沒有 QC 的情況下使用異常分數做風險判斷。
- 不要只靠 `json-repair` 判定資料可以使用。
- 不要讓同一病人的資料同時出現在 training 與 testing。
- 不要一次將完整 facts 與所有 QA 塞進 Mistral 7B prompt。

---

## 程式設計原則

1. 數值計算與語言生成分離。
2. 所有病例特定結論必須可追溯到 facts path。
3. 所有生成資料必須通過 validator。
4. 單病例失敗不可中斷整個 batch。
5. 所有長批次任務需可中斷後續跑。
6. 每次執行需保存 summary 與錯誤 log。
7. 保留 raw model output，方便 debug。
8. 輸入、輸出路徑使用 `pathlib.Path`。
9. JSON 使用 UTF-8 與 `ensure_ascii=False`。
10. 病例 ID 統一成 `patient0001` 到 `patient0100`。
11. 任何 risk、severity、diagnosis 相關欄位都需有 method、evidence、limitation 與 QC status。

---

## Codex 執行任務時的原則

Codex 在修改此專案時，應優先：

1. 先閱讀現有檔案與資料格式。
2. 不任意更改既有 JSON schema。
3. 若需要改 schema，需保持向後相容或提供 migration。
4. 不刪除 canonical QA。
5. 不把 LLM generated QA 當作唯一 ground truth。
6. 修改批次程式時保留 resume、retry 與 debug 功能。
7. 修改 medical wording 時保持保守，不擴張診斷能力。
8. 任何新增疾病功能都必須先確認是否有對應影像 evidence、ground truth，且可被目前資料驗證。
9. 優先完成可驗證、可展示、可寫進論文的功能。
10. 若不確定醫學結論，保留限制，不自行補充診斷。

---

## 專案當前一句話摘要

本專案將心臟 CT 分割結果轉換成可追溯的結構化事實，並使用 evidence-grounded LLM 產生技術型、病患友善型與安全限制型問答；目前重點是完成主動脈瓣鈣化與主動脈瓣狹窄風險的 QA 資料生成、批次處理、品質驗證、模型微調與中文 Demo。
