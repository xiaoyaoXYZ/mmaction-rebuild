"""Render selected original frames for audit inspection; never modify originals."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit_stage1/figures'
RAW=ROOT/'data/ucf101/rawframes'
OUT.mkdir(exist_ok=True,parents=True)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)

def sheet(name,title,rows):
    w,h=1040,56+len(rows)*220
    canvas=Image.new('RGB',(w,h),'white'); draw=ImageDraw.Draw(canvas)
    draw.text((12,12),title,font=font,fill='black')
    for i,(clip,frames) in enumerate(rows):
        y=52+i*220
        for j,k in enumerate(frames):
            path=RAW/clip/f'img_{k:05}.jpg'
            with Image.open(path) as im:
                im.thumbnail((328,184))
                canvas.paste(im,(12+j*342,y))
            draw.text((12+j*342,y+185),clip.split('/')[1]+f' #{k}',font=small,fill='black')
    canvas.save(OUT/name)

sheet('exact_overlap_examples.png','Exact-byte shared frames: paired rows; original content, no edits',[
 ('an_mopping/v_an_mopping_1',[111,113,116]),('an_mopping/v_an_mopping_2',[73,75,78]),
 ('cha_fang_you/v_cha_fang_you_1',[27,29,32]),('cha_fang_you/v_cha_fang_you_2',[144,146,149]),
 ('cha_fang_zuo/v_cha_fang_zuo_1',[81,87,94]),('cha_fang_zuo/v_cha_fang_zuo_2',[124,130,137])])
sheet('temporal_order_examples.png','Adjacent filename IDs: capture order and editing history are UNKNOWN',[
 ('cha1/v_cha1_1',[23,24,25]),('cha1/v_cha1_1',[34,35,36]),
 ('cha_fang_you/v_cha_fang_you_1',[26,27,28]),
 ('cha_fang_zuo/v_cha_fang_zuo_1',[80,81,82])])
sheet('near_match_examples.png','Perceptual candidates only: shared pixels/recording NOT established',[
 ('an_mopping/v_an_mopping_1',[91,92,110]),('an_mopping/v_an_mopping_2',[1,2,72]),
 ('cha1/v_cha1_1',[170,171,172]),('cha1/v_cha1_2',[1,2,3])])
print('Saved 3 evidence sheets')
