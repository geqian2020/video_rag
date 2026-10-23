"""合成术语图上的真实 RapidOCR 检查，不代替真实视频准确率。"""
import argparse
from pathlib import Path
import time
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from core.config import Settings
from core.io import write_json
from ocr.extract import load_engine
from eval.run_asr_eval import contains


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--font',default='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    p.add_argument('--output',default='data/public_benchmark/ocr_smoke')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if not Path(a.font).is_file():raise FileNotFoundError('请指定已安装的 TTF 字体')
    terms=['Qdrant','LangGraph','CrewAI','Whisper','BGE-M3','BM25','Qwen3','RRF']
    engine=load_engine(Settings());results=[]
    for name,size,blur in [('clear',28,0),('small_blurred',12,.5)]:
        im=Image.new('RGB',(720,300),'white');draw=ImageDraw.Draw(im)
        for i,term in enumerate(terms):
            draw.text((24+(i%2)*340,14+(i//2)*65),term,font=ImageFont.truetype(a.font,size),fill='black')
        if blur:im=im.filter(ImageFilter.GaussianBlur(blur))
        for scale in [1,2]:
            frame=im.resize((im.width*scale,im.height*scale),Image.Resampling.LANCZOS)
            path=root/f'{name}_{scale}.png';frame.save(path)
            start=time.monotonic();obs,_=engine(str(path));text=' '.join(r[1] for r in (obs or []))
            recognized=[t for t in terms if contains(text,t)]
            results.append({'fixture':name,'upscale':scale,'expected_terms':terms,'recognized_terms':recognized,
                            'term_recall':len(recognized)/len(terms),'ocr_text':text,'elapsed_seconds':time.monotonic()-start})
    write_json(root/'report.json',{'fixture_type':'synthetic English slide; actual OCR, not real-video accuracy','results':results})
    print(results)


if __name__=='__main__':main()
