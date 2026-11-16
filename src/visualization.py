"""Thin adapter over the existing SeeGround/PyTorch3D renderer."""
import sys,copy,json,hashlib,math,colorsys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PIL import Image,ImageDraw
import renderer as old
from scene_api import Geom

# Fixed display palette, indexed by the original LitePT 16-class IDs.
PALETTE=np.array([[112,128,144],[220,20,60],[255,127,80],[0,158,230],[233,150,70],[255,61,132],[255,243,64],[255,140,0],[175,90,210],[40,210,240],[140,140,140],[110,100,130],[180,150,150],[120,110,65],[170,170,180],[80,180,80]])/255

@dataclass
class Camera:
    origin:np.ndarray
    camera:object
    metadata:dict
    image_size:tuple

class Viz:
    def __init__(self,scene,out):self.scene=scene;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.views={}
    def box_style(self,cid):
        if cid in ['KEEP','D0','SOURCE'] or cid.startswith('DET:'):return (185,199,213),'grey baseline/context'
        index=self.scene.candidate_ids.index(cid)-1
        palette=[((55,220,155),'green'),((255,85,110),'red'),((255,174,55),'orange'),((186,136,255),'purple'),((65,185,255),'blue'),((245,220,65),'yellow')]
        if index<len(palette):return palette[index]
        rgb=tuple(int(255*x) for x in colorsys.hsv_to_rgb((.13+index*.381966)%1,.72,1.))
        return rgb,'RGB '+str(rgb)
    def make_camera(self,look_at,eye_direction,up=(0,0,1),projection='orthographic',span_m=12.,image_size=700):
        if projection!='orthographic':raise ValueError('Current reused renderer supports metric orthographic cameras')
        size=(image_size,image_size) if isinstance(image_size,int) else tuple(image_size)
        if size[0]!=size[1] or not 128<=size[0]<=1400:raise ValueError('Use a square camera, 128..1400 pixels')
        direction=np.asarray(eye_direction,dtype=float);direction/=np.linalg.norm(direction);distance=max(40.,float(span_m)*4)
        R,T=old.look_at_view_transform(eye=(tuple(direction*distance),),at=((0,0,0),),up=(tuple(up),),device='cpu')
        camera=old.FoVOrthographicCameras(device='cpu',R=R,T=T,znear=.1,zfar=distance*3,min_x=-span_m/2,max_x=span_m/2,min_y=-span_m/2,max_y=span_m/2)
        metadata=dict(look_at=np.asarray(look_at).tolist(),eye_direction=direction.tolist(),up=list(up),projection=projection,span_m=float(span_m),image_size=list(size))
        return Camera(np.array(look_at),camera,metadata,size)
    def preset_camera(self,name='main',points=None,box_ids=None,image_size=700):
        pts=self.scene.initial_points() if points is None else points;b=self.scene.box();origin=b.center
        xyz=pts.xyz-origin;inside=Geom.points_in_box(pts.xyz,b)
        # Empty real crops remain empty. Camera may use D0 corners for framing,
        # but no corner is ever inserted into the rendered point cloud.
        support=xyz if len(xyz) else old.corners(b.raw)-origin
        inside_arg=inside if len(xyz) else np.ones(8,bool)
        cameras,selection=old.choose_views(support,inside_arg,b.raw,origin,self.scene.sensor_origin-origin,'cpu')
        ids=(self.scene.candidate_ids if self.scene.task=='detection' else ['KEEP']) if box_ids is None else box_ids
        old.SIZE=(image_size,image_size)
        cameras,fit=old.fit_cameras(cameras,support,[self.scene.box(c).raw for c in ids],origin,'cpu')
        aliases={'BEV':'bev','Main':'main','complementary':'complementary'};name=aliases.get(name,name)
        if name not in cameras:raise ValueError('Preset main/complementary/bev')
        meta=dict(preset=name,look_at=origin.tolist(),selection=selection[name],span_m=2*fit[name]['half_extent_m'],image_size=[image_size,image_size],projection='orthographic',R=cameras[name].R.tolist(),T=cameras[name].T.tolist())
        return Camera(origin.copy(),cameras[name],meta,(image_size,image_size))
    def render(self,points,camera,box_ids=('KEEP',),color_mode='intensity',labels=None,highlight_ids=None,title='',tracks=None):
        pts=points;old.SIZE=camera.image_size;xyz=pts.xyz-camera.origin
        ids=np.asarray(pts.point_ids)
        if ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids):raise ValueError('Render unique original integer point IDs, without duplication')
        original=self.scene.points(point_ids=ids)
        if not np.array_equal(pts.xyz,original.xyz) or not np.array_equal(pts.intensity,original.intensity):
            raise ValueError('Renderer accepts unchanged current LiDAR global XYZ/intensity only. Fetch scene.points(point_ids=...) after numerical transformations.')
        if color_mode in ['semantic','semantics','labels']:
            labels=pts.semantic_labels if labels is None else np.asarray(labels)
            if len(labels)!=len(pts):raise ValueError('labels must align with the PointSet')
            allowed=np.array_equal(labels,original.semantic_labels)
            if not allowed and self.scene.task=='segmentation':allowed=any(np.array_equal(labels,self.scene.labels_for(ids,c)) for c in self.scene.candidate_ids[1:])
            if not allowed:raise ValueError('Semantic colors must show original S0 or an existing candidate preview; use highlight_ids for derived analysis sets')
            colors=PALETTE[labels]
        elif color_mode=='intensity':
            ref=self.scene.box();inside=Geom.points_in_box(pts.xyz,ref)
            values=pts.intensity.astype(float)
            lo,hi=np.percentile(values,[2,98]) if len(values) else (0.,1.)
            values=np.clip((values-lo)/max(hi-lo,1e-6),0,1)
            brightness=np.where(inside,.78+.22*values,.50+.26*values)
            colors=np.repeat(brightness[:,None],3,1)
        else:raise ValueError('color_mode is intensity or semantic')
        highlights=np.array([],dtype=np.int64) if highlight_ids is None else np.asarray(highlight_ids)
        if highlights.ndim!=1 or (len(highlights) and (highlights.dtype.kind not in 'iu' or not np.isin(highlights,ids).all())):
            raise ValueError('highlight_ids must be actual original IDs included in this rendered PointSet')
        highlights=highlights.astype(np.int64)
        if len(highlights):colors[np.isin(pts.point_ids,highlights)]=[1,.28,.15]
        if len(pts):im,depth,occupied=old.render(xyz,colors,camera.camera,'cpu',size=camera.image_size,radius_pixels=2.3)
        else:im=Image.new('RGB',camera.image_size,old.BG);occupied=0
        box_records=[]
        for cid in box_ids:
            b=self.scene.box(cid);rgb,color_name=self.box_style(cid);baseline=cid in ['KEEP','D0','SOURCE'] or cid.startswith('DET:');dashed=cid in ['KEEP','D0'] and len(box_ids)>1
            old.draw_box(im,camera.camera,b.raw,camera.origin,'cpu',rgb+(210 if baseline else 235,),width=1 if baseline else 2,dashed=dashed)
            box_records.append(dict(id=cid,box=b.raw,color_rgb=list(rgb),color_name=color_name,line_style='dashed' if dashed else 'solid'))
        if tracks:
            known={json.dumps(r,sort_keys=True) for r in self.scene.track_history()}
            if any(json.dumps(r,sort_keys=True) not in known for r in tracks):
                raise ValueError('Use unchanged records from scene.track_history(); do not invent or move prior states')
            # These are explicitly labelled prior states, never claimed matches.
            draw=ImageDraw.Draw(im)
            for record in tracks:
                c=np.array(record['box'][:3]);uv,_=old.project(camera.camera,[c-camera.origin],'cpu',size=camera.image_size);x,y=uv[0]
                draw.ellipse((x-3,y-3,x+3,y+3),outline=(80,235,150),width=2);draw.text((x+5,y),'prior '+str(record['track_id']),fill=(80,235,150))
        old.scale_bar(im,camera.camera,'cpu')
        title=title or 'Current LiDAR / '+color_mode
        display_title=title
        while old.font(24,True).getlength(display_title)>im.width-40:display_title=display_title[:-4]+'...'
        framed=old.panel(im,display_title,'Real points: '+str(len(pts))+' | global metres | no GT')
        if box_records:
            legend=Image.new('RGB',(framed.width,26*len(box_records)+6),old.BG);d=ImageDraw.Draw(legend)
            for j,b in enumerate(box_records):
                cid=b['id'];color=tuple(b['color_rgb'])
                d.line((8,j*26+15,30,j*26+15),fill=color,width=2);d.text((39,j*26+3),cid,font=old.font(16),fill=color)
            combined=Image.new('RGB',(framed.width,framed.height+legend.height),old.BG);combined.paste(framed,(0,0));combined.paste(legend,(0,framed.height));framed=combined
        if color_mode in ['semantic','semantics','labels']:
            present=np.unique(labels);legend=Image.new('RGB',(framed.width,32*math.ceil(len(present)/3)+6),old.BG);d=ImageDraw.Draw(legend)
            for j,k in enumerate(present):
                x=(j%3)*int(framed.width/3);y=(j//3)*32;d.rectangle((x+5,y+7,x+19,y+21),fill=tuple((PALETTE[k]*255).astype(int)));d.text((x+24,y+5),self.scene.class_names[str(k)],font=old.font(13),fill=old.WHITE)
            combined=Image.new('RGB',(framed.width,framed.height+legend.height),old.BG);combined.paste(framed,(0,0));combined.paste(legend,(0,framed.height));framed=combined
        vid='V'+str(len(self.views)+1);path=self.out/(vid+'.png');framed.save(path)
        support_path=self.out/(vid+'.points.npz');np.savez_compressed(support_path,point_ids=ids,highlight_ids=highlights)
        meta=dict(view_id=vid,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),camera=camera.metadata,points=len(pts),point_ids_sha256=hashlib.sha256(pts.point_ids.astype(np.int64).tobytes()).hexdigest(),point_ids_file=str(support_path),highlight_ids=highlights.tolist(),color_mode=color_mode,boxes=box_records,occupied_pixels=occupied,track_records=tracks or [],tracks='explicit unconfirmed prior states' if tracks else 'not overlaid',title=title)
        (self.out/(vid+'.json')).write_text(json.dumps(meta));self.views[vid]=meta;return meta
    def load_view(self,view_id):return self.views[view_id]
