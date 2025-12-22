
import json

results = [
    {
        "question": "像我那次就特别出丑，我感觉",
        "true_label": "unfinished",
        "pred_label": "<think>句子“像我那次就特别出丑，我感觉”缺少后续的宾语或补语来完整表达“我感觉”后的情况或结果，因此属于动词悬空，按照规则中的“动词悬空”判断为unfinished。</think> <solution>unfinished</solution>"
    },
    {
        "question": "不知道上了幼儿园没有，那会儿是我们那个村里面",
        "true_label": "unfinished",
        "pred_label": "<think>句子“不知道上了幼儿园没有，那会儿是我们那个村里面”表达不完整，缺乏清晰的语义关系，如“那会儿”后应有进一步的说明或补充，缺少必要的宾语或补语，属于语义不完整。</think> <solution>unfinished</solution>"
    },
    {
        "question": "但小小的时候拿那个泥做过，当时",
        "true_label": "unfinished",
        "pred_label": "<think>句子“但小小的时候拿那个泥做过，当时”缺少后续信息来说明具体做了什么，因此语义不完整。</think> <solution>unfinished</solution>"
    },
    {
        "question": "哇听你讲我都觉得好浪漫啊",
        "true_label": "finished",
        "pred_label": "<think>句子“哇听你讲我都觉得好浪漫啊”中，虽然有口语化表达“啊”，但核心语义结构“我（主语）觉得好浪漫（谓语+宾语）”完整，符合完整的陈述结构规则。</think> <solution>finished</solution>"
    },
    {
        "question": "是的可惜不能和女朋友一起，这是最大的遗憾",
        "true_label": "finished",
        "pred_label": "<think>“是的可惜不能和女朋友一起”是一个完整的陈述句，表达了明确的评价和情感状态，符合“结果导向”规则中的明确表达预期结果、评价或询问结果，因此判断为finished。</think> <solution>finished</solution>"
    },
    {
        "question": "哎，你能不能不想着你女朋友啊？",
        "true_label": "finished",
        "pred_label": "<think>句子“哎，你能不能不想着你女朋友啊？”是一个完整的疑问句，使用了疑问词“能不能”和完整的动宾结构“想着你女朋友”，结尾使用了疑问语气词“啊”，符合“疑问完整”原则。</think> <solution>finished</solution>"
    },
    {
        "question": "鼓楼，是那种两层楼的吗？",
        "true_label": "finished",
        "pred_label": "<think>句子“鼓楼，是那种两层楼的吗？“是一个完整的疑问句，包括疑问词“什么“、谓语“是“、宾语“那种两层楼“以及疑问语气词“吗“。根据“疑问完整“规则，这是一个完整的疑问结构。</think> <solution>finished</solution>"
    },
    {
        "question": "啊，那倒不是",
        "true_label": "finished",
        "pred_label": "<think>句子“啊，那倒不是”中断，没有完整的谓语和宾语结构，语义不完整，无法理解完整意图。</think> <solution>unfinished</solution>"
    },
    {
        "question": "现在需要静一静",
        "true_label": "wait",
        "pred_label": "<think>句中使用了“现在需要静一静”，表达了用户当前需要暂停对话，符合wait信号词的定义（明确拒绝继续或延迟处理），因此判断为wait。</think> <solution>wait</solution>"
    },
    {
        "question": "闭嘴，立刻停止",
        "true_label": "wait",
        "pred_label": "<think>句子“闭嘴，立刻停止”含有明确的停止对话的请求，属于情绪性拒绝，因此按照规则“情绪性拒绝”进行判断。</think> <solution>wait</solution>"
    },
    {
        "question": "请现在保持安静",
        "true_label": "wait",
        "pred_label": "<think>句子“请现在保持安静”表达了用户希望对方暂时保持安静的请求，属于wait中的“明确拒绝继续”，因此判断为wait。</think> <solution>wait</solution>"
    }
]

print("="*80)
print("TurnLiteV4 CoT 推理结果展示 (Evaluation Case Study)")
print("="*80)

for item in results:
    print(f"\nQuestion: {item['question']}")
    print(f"True Label: {item['true_label']}")
    print(f"Prediction (CoT):\n{item['pred_label']}")
    print("-" * 40)

print("\n展示完毕。")
