"""Actual-point PyTorch3D renderer extracted from the experiment.
create_point_cloud is adapted from SeeGround (Apache-2.0); see third_party.
"""
import math,sys
from pathlib import Path
import matplotlib
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from pytorch3d.renderer import PointsRenderer,PointsRasterizer,PointsRasterizationSettings,AlphaCompositor,FoVOrthographicCameras,look_at_view_transform
from pytorch3d.structures import Pointclouds
SIZE=(700,700)
POINT_RADIUS=2.5
BG=(9,18,29)
WHITE=(225,236,249)
EDGES=[(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count('1')==1]
def create_point_cloud(scan_pc, device):
    """
    Create a point cloud from scan data.

    Args:
        scan_pc (np.ndarray): The scan data containing points and colors.
        device (str): The device to use for computation.

    Returns:
        Pointclouds: The created point cloud.
    """
    points = torch.tensor(scan_pc[:, :3], dtype=torch.float32)
    colors = torch.tensor(scan_pc[:, 3:], dtype=torch.float32)
    point_cloud = Pointclouds(points=[points], features=[colors]).to(device)
    return point_cloud

def font(size,bold=False):
    name='DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    try:return ImageFont.truetype(str(Path(sys.modules['matplotlib'].__file__).parent/'mpl-data/fonts/ttf'/name),size)
    except OSError:return ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono'+('-Bold' if bold else '')+'.ttf',size)


def rotation(box):
    q=box['rotation'];return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()


def corners(box):
    w,l,h=box['size'];s=np.array([[1 if i>>k&1 else -1 for k in range(3)] for i in range(8)])
    return (s*np.array([l,w,h])/2)@rotation(box).T+box['translation']


def orthographic(azimuth,elevation,device,half_extent=1.,distance=40.,up=None):
    if up is None:
        az=math.radians(azimuth);el=math.radians(elevation)
        eye=(distance*math.cos(el)*math.cos(az),distance*math.cos(el)*math.sin(az),distance*math.sin(el))
        up=(0.,0.,1.)
    else:eye=(0.,0.,distance)
    R,T=look_at_view_transform(eye=(eye,),at=((0.,0.,0.),),up=(tuple(up),),device=device)
    # Identical X/Y scale preserves physical lengths; no per-axis stretching.
    return FoVOrthographicCameras(device=device,R=R,T=T,znear=.1,zfar=distance*3,
        min_x=-half_extent,max_x=half_extent,min_y=-half_extent,max_y=half_extent)


def view_xy(camera,points,device):
    a=torch.tensor(points,dtype=torch.float32,device=device)
    return camera.get_world_to_view_transform().transform_points(a).detach().cpu().numpy()[:,:2]


def choose_views(points,inside,box,origin,sensor,device):
    """Direction selection uses only D0, current points and sensor position."""
    yaw=math.degrees(math.atan2(rotation(box)[1,0],rotation(box)[0,0]))
    az_observed=math.degrees(math.atan2(sensor[1],sensor[0]))
    d0=corners(box)-origin;distance=max(40.,8*float(max(box['size'])))
    trials=[]
    for offset in range(-40,41,5):
        az=az_observed+offset
        rel=math.radians(az-yaw)
        # Avoid views down either box axis, while remaining on the observed side.
        side=min(abs(math.sin(rel)),abs(math.cos(rel)))
        if side<math.sin(math.radians(25)):continue
        cam=orthographic(az,30.,device,distance=distance)
        xy=view_xy(cam,points,device);bxy=view_xy(cam,d0,device)
        half=max(float(np.abs(xy).max()),float(np.abs(bxy).max()))/0.95
        sample=xy[inside] if inside.any() else xy
        cells=np.floor(sample/(2*half)*200).astype(np.int64)
        distinct=len(np.unique(cells,axis=0))/len(sample)
        coverage=float(np.ptp(bxy,axis=0).max()/(2*half))
        score=distinct+.65*coverage-.08*abs(offset)/40
        trials.append(dict(azimuth_deg=az,offset_from_lidar_deg=offset,elevation_deg=30.,
            distinct_projected_cells_fraction=distinct,d0_extent_fraction=coverage,score=score))
    if not trials:
        trials=[dict(azimuth_deg=az_observed+30.,offset_from_lidar_deg=30.,elevation_deg=30.,score=0.)]
    best=max(trials,key=lambda x:x['score']);az=best['azimuth_deg']
    # A ninety-degree complementary direction, on the side closer to the sensor.
    alternatives=[az-90.,az+90.]
    complementary=min(alternatives,key=lambda a:abs((a-az_observed+180)%360-180))
    long=rotation(box)[:2,0];long=long/np.linalg.norm(long)
    cameras=dict(main=orthographic(az,30.,device,distance=distance),
        complementary=orthographic(complementary,20.,device,distance=distance),
        bev=orthographic(0.,90.,device,distance=distance,up=np.r_[long,0.]))
    metadata=dict(selection_uses='D0, actual current crop points, and LiDAR origin only; no candidates or GT',
        lidar_side_azimuth_deg=az_observed,d0_long_axis_yaw_deg=yaw,
        main=best,complementary=dict(azimuth_deg=complementary,elevation_deg=20.,azimuth_separation_deg=90.),
        bev=dict(elevation_deg=90.,up_global=np.r_[long,0.].tolist(),long_axis_aligned_vertical=True),
        distance_m=distance,direction_trials=trials)
    return cameras,metadata


def fit_cameras(cameras,points,boxes,origin,device):
    """Fit once to the unchanged crop and ALL boxes; share main/BEV scale."""
    support=np.concatenate([points]+[corners(b)-origin for b in boxes])
    halves={}
    for key,cam in cameras.items():
        xy=view_xy(cam,support,device)
        d0=view_xy(cam,corners(boxes[0])-origin,device)
        # Aim for 78% occupancy, but never clip a retained point or a box.
        halves[key]=max(float(np.abs(xy).max())/.95,float(np.ptp(d0,axis=0).max())/(2*.78))
    halves['main']=halves['bev']=max(halves['main'],halves['bev'])
    info={}
    for key,cam in list(cameras.items()):
        half=halves[key]
        cameras[key]=FoVOrthographicCameras(device=device,R=cam.R,T=cam.T,znear=cam.znear,zfar=cam.zfar,
            min_x=-half,max_x=half,min_y=-half,max_y=half)
        d0=view_xy(cam,corners(boxes[0])-origin,device)
        info[key]=dict(half_extent_m=half,meters_per_pixel=2*half/SIZE[0],
            d0_extent_fraction=float(np.ptp(d0,axis=0).max()/(2*half)),
            all_points_and_all_boxes_fit=True)
    return cameras,info


def render(points,colors,camera,device,size=SIZE,radius_pixels=POINT_RADIUS):
    cloud=create_point_cloud(np.concatenate([points,colors],axis=1),device)
    rasterizer=PointsRasterizer(cameras=camera,raster_settings=PointsRasterizationSettings(
        image_size=size,radius=2*radius_pixels/min(size),points_per_pixel=10,bin_size=0))
    renderer=PointsRenderer(rasterizer=rasterizer,compositor=AlphaCompositor(background_color=np.array(BG)/255))
    with torch.no_grad():
        rgb=renderer(cloud)[0,...,:3].clamp(0,1).cpu().numpy()
        fragments=rasterizer(cloud)
        # Exclude empty fragment slots (-1); their values are not physical depth.
        depth=fragments.zbuf[0].cpu().numpy();depth=np.where(depth>0,depth,np.inf).min(axis=-1)
        visible=(fragments.idx[0,...,0]>=0).cpu().numpy()
    assert np.isfinite(rgb).all() and visible.any()
    return Image.fromarray(np.rint(rgb*255).astype(np.uint8)),depth,int(visible.sum())


def project(camera,xyz,device,size=None):
    if size is None:size=SIZE
    a=torch.tensor(np.asarray(xyz).reshape(-1,3),dtype=torch.float32,device=device)
    with torch.no_grad():
        screen=camera.transform_points_screen(a[None],image_size=(size,))[0].cpu().numpy()
        z=camera.get_world_to_view_transform().transform_points(a)[...,2].cpu().numpy()
    return screen[:,:2],z


def line(draw,p,q,color,width=2,dashed=False):
    p=np.asarray(p);q=np.asarray(q)
    if not np.isfinite([p,q]).all():return
    if not dashed:draw.line([tuple(p),tuple(q)],fill=color,width=width);return
    length=np.linalg.norm(q-p)
    for start in np.arange(0,length,10):
        a=p+(q-p)*(start/max(length,1));b=p+(q-p)*(min(start+5,length)/max(length,1))
        draw.line([tuple(a),tuple(b)],fill=color,width=width)


def draw_box(im,camera,box,origin,device,color,width=1,dashed=False):
    draw=ImageDraw.Draw(im,'RGBA');p,z=project(camera,corners(box)-origin,device)
    for a,b in EDGES:
        assert z[a]>0 and z[b]>0
        line(draw,p[a],p[b],color,width,dashed)
    center,_=project(camera,[np.array(box['translation'])-origin],device)
    x,y=center[0];draw.line((x-4,y,x+4,y),fill=color,width=1);draw.line((x,y-4,x,y+4),fill=color,width=1)
    return p.tolist()


def scale_bar(im,camera,device):
    # A metric unit in the camera plane, not a foreshortened world-X unit.
    right=camera.R[0,:,0].detach().cpu().numpy()
    p,_=project(camera,[np.zeros(3),right],device);length=float(np.linalg.norm(p[1]-p[0]))
    d=ImageDraw.Draw(im);x=23;y=SIZE[0]-25
    d.line((x,y,x+length,y),fill=WHITE,width=2)
    d.text((x,y-26),'1 m',font=font(18),fill=WHITE)


def panel(image,title,subtitle):
    w,h=image.size;out=Image.new('RGB',(w,h+88),(16,29,44));out.paste(image,(0,88))
    d=ImageDraw.Draw(out);d.text((20,12),title,font=font(24,True),fill=WHITE)
    d.text((20,49),subtitle,font=font(17),fill=(175,193,211))
    return out

