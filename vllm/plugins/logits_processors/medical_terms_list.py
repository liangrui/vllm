# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""扩展的医学术语列表"""

# 更完整的中文医学术语列表
EXTENDED_CHINESE_MEDICAL_TERMS = [
    # 生命体征
    "体温", "脉搏", "血压", "心率", "呼吸频率", "血氧饱和度",
    "血糖", "胆固醇", "甘油三酯", "尿酸", "肌酐", "尿素氮",
    
    # 检查和影像
    "心电图", "X光", "CT扫描", "MRI", "核磁共振", "B超",
    "超声检查", "胃镜", "肠镜", "血常规", "尿常规", "生化检查",
    "PET-CT", "脑电图", "肌电图", "骨密度",
    
    # 治疗和手术
    "手术", "治疗", "化疗", "放疗", "靶向治疗", "免疫治疗",
    "诊断", "处方", "药物", "剂量", "注射", "输液",
    "手术切除", "微创手术", "激光手术", "移植", "透析",
    
    # 科室
    "内科", "外科", "儿科", "妇科", "产科", "骨科",
    "神经科", "神经内科", "神经外科", "心血管内科", "心脏外科",
    "呼吸内科", "消化内科", "肾内科", "血液科", "内分泌科",
    "皮肤科", "眼科", "耳鼻喉科", "口腔科", "肿瘤科",
    "急诊科", "重症监护室", "ICU", "康复科", "中医科",
    
    # 医疗人员
    "医生", "医师", "护士", "护师", "药师", "技师",
    "教授", "专家", "主任", "副主任", "主治医生",
    "住院医师", "实习生", "患者", "病人", "家属",
    
    # 医疗场所和流程
    "医院", "诊所", "门诊", "住院", "出院", "转院",
    "挂号", "候诊", "就诊", "病历", "病史", "医嘱",
    "查房", "交接班", "会诊", "转诊",
    
    # 疾病和症状
    "感染", "炎症", "病毒", "细菌", "真菌", "寄生虫",
    "肿瘤", "癌症", "良性肿瘤", "恶性肿瘤", "转移",
    "糖尿病", "高血压", "冠心病", "心肌梗塞", "脑梗塞",
    "脑出血", "哮喘", "肺炎", "支气管炎", "胃炎",
    "胃溃疡", "肝炎", "肝硬化", "肾炎", "肾结石",
    "贫血", "白血病", "淋巴瘤", "骨折", "脱位",
    "关节炎", "颈椎病", "腰椎间盘突出", "中风", "癫痫",
    "头痛", "偏头痛", "发热", "咳嗽", "咳痰",
    "呼吸困难", "胸痛", "腹痛", "腹泻", "便秘",
    "恶心", "呕吐", "黄疸", "水肿", "皮疹",
    "瘙痒", "疼痛", "麻木", "乏力", "失眠",
    
    # 药物和疫苗
    "疫苗", "抗体", "抗原", "抗生素", "消炎药", "止痛药",
    "退烧药", "降压药", "降糖药", "激素", "维生素",
    "中药", "西药", "中成药", "注射剂", "片剂",
    "胶囊", "药膏", "眼药水", "滴鼻剂",
    
    # 生理和解剖
    "免疫", "免疫系统", "遗传", "基因", "染色体", "DNA",
    "心脏", "肺", "肝", "脾", "肾", "胃",
    "肠", "脑", "神经", "血管", "动脉", "静脉",
    "肌肉", "骨骼", "关节", "皮肤", "眼睛", "耳朵",
    "鼻子", "喉咙", "口腔", "牙齿",
    
    # 康复和保健
    "康复", "康复训练", "理疗", "针灸", "推拿",
    "护理", "保健", "体检", "健康检查", "复查",
    "随访", "预防", "防疫", "消毒", "隔离",
    
    # 其他
    "救护车", "急救", "抢救", "病危", "病重",
    "治愈", "好转", "恶化", "并发症", "后遗症",
    "医保", "公费医疗", "自费", "费用", "账单"
]

# 更完整的英文医学术语列表
EXTENDED_ENGLISH_MEDICAL_TERMS = [
    # Vital signs
    "temperature", "pulse", "blood pressure", "heart rate", 
    "respiratory rate", "oxygen saturation",
    "blood sugar", "glucose", "cholesterol", "triglycerides",
    "uric acid", "creatinine", "urea nitrogen",
    
    # Examinations and imaging
    "ECG", "EKG", "electrocardiogram", "X-ray", "CT scan",
    "MRI", "magnetic resonance imaging", "ultrasound", "sonogram",
    "endoscopy", "gastroscopy", "colonoscopy", "blood test",
    "urinalysis", "biochemistry", "PET-CT", "EEG",
    "electroencephalogram", "EMG", "electromyography",
    "bone density", "DEXA scan",
    
    # Treatments and surgery
    "surgery", "operation", "treatment", "chemotherapy",
    "radiation therapy", "radiotherapy", "targeted therapy",
    "immunotherapy", "diagnosis", "prescription", "medication",
    "drug", "dosage", "injection", "infusion", "IV",
    "resection", "excision", "minimally invasive surgery",
    "laser surgery", "transplant", "transplantation",
    "dialysis", "hemodialysis", "peritoneal dialysis",
    
    # Departments
    "internal medicine", "surgery", "pediatrics", "gynecology",
    "obstetrics", "orthopedics", "neurology", "neurosurgery",
    "cardiology", "cardiovascular", "cardiothoracic surgery",
    "pulmonology", "respiratory", "gastroenterology",
    "nephrology", "hematology", "endocrinology", "dermatology",
    "ophthalmology", "ENT", "otolaryngology", "dentistry",
    "oncology", "emergency", "emergency room", "ER",
    "intensive care unit", "ICU", "rehabilitation", "physical therapy",
    "traditional Chinese medicine", "TCM",
    
    # Medical personnel
    "doctor", "physician", "nurse", "nurse practitioner",
    "pharmacist", "technician", "technologist", "professor",
    "specialist", "attending", "consultant", "resident",
    "intern", "patient", "family", "caregiver",
    
    # Healthcare facilities and processes
    "hospital", "clinic", "outpatient", "inpatient",
    "admission", "discharge", "transfer", "appointment",
    "registration", "waiting room", "consultation", "medical record",
    "chart", "medical history", "doctor's order", "rounds",
    "handover", "consult", "referral",
    
    # Diseases and symptoms
    "infection", "inflammation", "virus", "bacteria", "fungus",
    "parasite", "tumor", "cancer", "malignancy", "benign",
    "metastasis", "diabetes", "hypertension", "high blood pressure",
    "coronary artery disease", "myocardial infarction", "heart attack",
    "stroke", "cerebrovascular accident", "hemorrhage", "bleeding",
    "asthma", "pneumonia", "bronchitis", "gastritis", "stomach ulcer",
    "peptic ulcer", "hepatitis", "cirrhosis", "nephritis",
    "kidney stones", "renal calculi", "anemia", "leukemia",
    "lymphoma", "fracture", "dislocation", "arthritis",
    "cervical spondylosis", "herniated disc", "seizure",
    "epilepsy", "headache", "migraine", "fever", "cough",
    "sputum", "shortness of breath", "dyspnea", "chest pain",
    "abdominal pain", "diarrhea", "constipation", "nausea",
    "vomiting", "jaundice", "edema", "swelling", "rash",
    "pruritus", "itching", "pain", "numbness", "fatigue",
    "insomnia",
    
    # Medications and vaccines
    "vaccine", "vaccination", "antibody", "antigen",
    "antibiotic", "anti-inflammatory", "analgesic", "painkiller",
    "antipyretic", "antihypertensive", "hypoglycemic", "steroid",
    "hormone", "vitamin", "supplement", "herbal medicine",
    "Western medicine", "traditional medicine", "pill", "tablet",
    "capsule", "ointment", "cream", "eye drops", "nasal spray",
    
    # Physiology and anatomy
    "immune", "immune system", "genetic", "gene", "chromosome",
    "DNA", "heart", "lung", "liver", "spleen", "kidney",
    "stomach", "intestine", "brain", "nerve", "blood vessel",
    "artery", "vein", "muscle", "bone", "joint", "skin",
    "eye", "ear", "nose", "throat", "mouth", "tooth",
    
    # Rehabilitation and health
    "rehabilitation", "physical therapy", "physiotherapy",
    "occupational therapy", "acupuncture", "massage", "tui na",
    "nursing", "health care", "medical examination", "check-up",
    "physical", "follow-up", "prevention", "epidemic prevention",
    "disinfection", "isolation", "quarantine",
    
    # Other
    "ambulance", "emergency", "first aid", "resuscitation",
    "critical", "seriously ill", "cure", "recovery",
    "improvement", "deterioration", "complication", "sequela",
    "health insurance", "medical insurance", "self-pay",
    "expense", "bill", "cost"
]
