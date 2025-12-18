from segment_anything import sam_model_registry
from demo import BboxPromptDemo
MedSAM_CKPT_PATH = "C:/Users/Administrator/Desktop/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth"
# MedSAM_CKPT_PATH = "D:\pth\sam_vit_h_4b8939.pth"
device = "cuda:0"
medsam_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
medsam_model = medsam_model.to(device)
medsam_model.eval()
# abdomen CT

img = "C:/Users/Administrator/Desktop/MedSAM-main/98a9f8e5b4f7cdeeaa0f07c09216c8c2.png"
bbox_prompt_demo = BboxPromptDemo(medsam_model)
bbox_prompt_demo.show(img)







