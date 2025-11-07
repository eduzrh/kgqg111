import json

def extract_qa_pairs(input_file, output_file):
   """
   从输入的jsonl文件中提取question和answer，组合成指定格式并保存到输出文件
   
   Args:
       input_file: 输入文件路径，例如 '1.jsonl'
       output_file: 输出文件路径，例如 '2.jsonl'
   """
   with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
       
       for line in infile:
           # 跳过空行
           if not line.strip():
               continue
           
           try:
               # 解析输入的JSON行
               data = json.loads(line)
               
               # 提取optimized_text字段
               optimized_text = data.get('optimized_text', '')
               
               if not optimized_text:
                   continue
               
               # 解析optimized_text中的JSON内容
               # 去除markdown代码块标记
               optimized_text = optimized_text.strip()
               if optimized_text.startswith('```json'):
                   optimized_text = optimized_text[7:]
               if optimized_text.endswith('```'):
                   optimized_text = optimized_text[:-3]
               
               optimized_data = json.loads(optimized_text.strip())
               
               # 提取qa_pairs
               qa_pairs = optimized_data.get('qa_pairs', [])
               
               # 遍历每个QA对
               for qa in qa_pairs:
                   question = qa.get('question', '')
                   answer = qa.get('answer', '')
                   
                   # 跳过空的question或answer
                   if not question or not answer:
                       continue
                   
                   # 组合成指定格式的文本
                   text = f"问题：{question}\n答案：{answer}"
                   
                   # 构建输出的JSON对象
                   output_data = {"text": text}
                   
                   # 写入输出文件
                   outfile.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                   
           except json.JSONDecodeError as e:
               print(f"JSON解析错误: {e}")
               continue
           except Exception as e:
               print(f"处理错误: {e}")
               continue

if __name__ == "__main__":
   # 使用示例
   input_file = '1.jsonl'
   output_file = '2.jsonl'
   
   extract_qa_pairs(input_file, output_file)
   print(f"处理完成！结果已保存到 {output_file}")
