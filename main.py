import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import kaolin as kal
import clip
import torchvision
import torchvision.transforms as transforms
import copy
import random
from pathlib import Path
from tqdm import tqdm

# parameters
root_path = '/content/drive/MyDrive/project/implementation/codes'
exp_name = 'candy horse'
n_iter = 750
progressive_encoding = True
width = 256
depth = 4
input_dim = 3
sigma = 6.0
lr = 0.0005
lr_decay = 0.9
standardize = True
obj_path = root_path + "/data/source_meshes/horse.obj"  
prompt = "a 3D rendering of a horse made of colorful candy"
out_dir = root_path + "/result/" + exp_name
frontview_center = [1.96349, 0.6283]
frontview_std = 4
radi = 2

# seeding
seed = 30
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.device(device)
    
# transforms
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(1, 1)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

norm_augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.1, 0.1)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

displacement_augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.1, 0.1)),
    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])
    
class fourier_feature(nn.Module):
    def __init__(self, input_channel, map_size, sigma=5):
        super().__init__()
        self.input_channel = input_channel
        self.map_size = map_size
        self.sigma = sigma
        self.B = self.create_matrix()
        
    def create_matrix(self):
        B = torch.randn((self.input_channel, self.map_size)) * self.sigma
        B = sorted(B, key=lambda x: torch.norm(x, p=2))
        B = torch.stack(B)
        return B    
        
    def forward(self, x):
        self.B = self.B.to(device)
        out = 2 * np.pi * torch.matmul(x, self.B)
        out = torch.cat([x, torch.sin(out), torch.cos(out)], dim=1)
        return out
        
class progessive_encoding(nn.Module):
    def __init__(self, map_size, n_iter, dim=3):
        super(progessive_encoding, self).__init__()
        self._t = 0
        self.n = map_size
        self.T = n_iter
        self.d = dim
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)], device=device)

    def forward(self, x):
        alpha = ((self._t - self._tau * self.indices) / self._tau).clamp(0, 1).repeat(2)  # no need to reduce d or to check cases
        alpha = torch.cat([torch.ones(self.d, device=device), alpha], dim=0)
        self._t += 1
        return x * alpha
        
class neural_style_field(nn.Module):
    def __init__(self, width, depth, sigma=5, pos_encode=True, n_iter=1500, input_dim=3):
        super(neural_style_field, self).__init__()
        
        self.pe_layer = progessive_encoding(map_size=width, n_iter=n_iter, dim=input_dim)
        
        if pos_encode:
            self.base_layer = nn.Sequential(fourier_feature(input_dim, width, sigma),
                                            self.pe_layer,
                                            nn.Linear(width * 2 + input_dim, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU())
        else :
            self.base_layer = nn.Sequential(fourier_feature(input_dim, width, sigma),
                                            nn.Linear(width * 2 + input_dim, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU(),
                                            nn.Linear(width, width),
                                            nn.ReLU())
                   
        self.mlp_color = nn.Sequential(nn.Linear(width, width),
                                       nn.ReLU(),
                                       nn.Linear(width, width),
                                       nn.ReLU(),
                                       nn.Linear(width, 3))
                             
                             
        self.mlp_norm = nn.Sequential(nn.Linear(width, width),
                                       nn.ReLU(),
                                       nn.Linear(width, width),
                                       nn.ReLU(),
                                       nn.Linear(width, 1))                                         
                                       
    def forward(self, x):
        x = self.base_layer(x)
        color = self.mlp_color(x)
        normal = self.mlp_norm(x)
        color = F.tanh(color) / 2
        normal = F.tanh(normal) * 0.1
        
        return color, normal
        
# mesh class of obj using kaolin
class obj_mesh():
    def __init__(self, obj_path, color = torch.Tensor([0.5, 0.5, 0.5])):
        mesh = kal.io.obj.import_mesh(obj_path, with_normals=True)
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.vertice_normals = mesh.vertex_normals
        self.face_normals = mesh.face_normals

        self.vertices = self.vertices.to(device)
        self.faces = self.faces.to(device)
        self.vertice_normals = F.normalize(self.vertice_normals.to(device).float())
        self.face_normals = F.normalize(self.face_normals.to(device).float())

        # set texture 
        self.texture = torch.zeros(224, 224, 3).unsqueeze(0).to(device)
        self.texture[:, :, :] = color
        self.texture = self.texture.permute(0, 3, 1, 2)

        # set face attributes
        self.face_attributes = torch.zeros(self.faces.shape[0], 3, 3).unsqueeze(0).to(device)
        self.face_attributes[:, :, :] = color
        
def camera_view(elevation, azimuth):
    x_coord = radi * torch.cos(elevation) * torch.sin(azimuth)
    y_coord = radi * torch.sin(elevation)
    z_coord = radi * torch.cos(elevation) * torch.sin(azimuth)

    position = torch.Tensor([[x_coord, y_coord, z_coord]])
    direction = torch.Tensor([[0.0, 1.0, 0.0]])

    return kal.render.camera.generate_transformation_matrix(position, -position, direction) 
    

class obj_renderer():
    def __init__(self, mesh='sample.obj',
                 lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 camera=kal.render.camera.generate_perspective_projection(np.pi / 3).to(device),
                 dim=(224, 224)):

        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim

    def render_front_views(self, mesh, n_views, background):
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        elev = torch.cat((torch.tensor([frontview_center[1]]), torch.randn(n_views - 1) * np.pi / frontview_std + frontview_center[1]))
        azim = torch.cat((torch.tensor([frontview_center[0]]), torch.randn(n_views - 1) * 2 * np.pi / frontview_std + frontview_center[0]))
        images = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device='cuda')
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(n_views):
            camera_transform = camera_view(elev[i], azim[i]).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])

            image_features, mask = image_features

            image = torch.clamp(image_features, 0.0, 1.0)
            image_normals = face_normals[:, face_idx].squeeze(0)
            image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
            image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
            image = torch.clamp(image, 0.0, 1.0)

            background_mask = torch.zeros(image.shape).to(device)
            mask = mask.squeeze(-1)
            assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
            background_mask[torch.where(mask == 0)] = background
            image = torch.clamp(image + background_mask, 0., 1.)
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        return images, elev, azim

def mesh_normalizer(mesh):
    mesh_vertices = mesh.vertices
    shift = torch.mean(mesh_vertices, dim=0)
    scale = torch.max(torch.norm(mesh_vertices-shift, p=2, dim=1))
    mesh.vertices = (mesh_vertices - shift) / scale
    return mesh
    
if __name__ == '__main__':
    # define the network
    net = neural_style_field(width=width,
                             depth=depth,
                             sigma=sigma,
                             pos_encode=True,
                             n_iter=n_iter,
                             input_dim=input_dim)
    net = net.to(device)
    print(net)
    optim = torch.optim.Adam(net.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=lr_decay) 
    
    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    render = obj_renderer()

    mesh = obj_mesh(obj_path)
    mesh = mesh_normalizer(mesh)

    net_input = copy.deepcopy(mesh.vertices)
    vertices = copy.deepcopy(net_input)
    color_now = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)
    background = torch.Tensor([1, 1, 1]).to(device) # white
    
    # clip encode prompt
    adj_prompt = " ".join(prompt)
    clip_encoded_prompt = clip_model.encode_text(clip.tokenize([adj_prompt]).to(device))
    
    # train
    optim.zero_grad()
    for iter in tqdm(range(n_iter)):
        this_mesh = mesh
        net_input = net_input.to(device)
        pred_color, pred_normal = net(net_input)
        this_mesh.vertices = vertices + this_mesh.vertice_normals * pred_normal
        this_mesh.face_attributes = color_now + kal.ops.mesh.index_vertices_by_faces(pred_color.unsqueeze(0),
                                                                                        this_mesh.faces)
        this_mesh = mesh_normalizer(this_mesh)
        rendered_images, _, _ = render.render_front_views(this_mesh, 
                                                          n_views=5,
                                                          background=background)
    
        loss = 0.0
        clip_encoded_render = clip_model.encode_image(augment_transform(rendered_images))
        
        if clip_encoded_prompt.shape[0] > 1:
            loss = loss - torch.cosine_similarity(torch.mean(clip_encoded_render, dim=0), torch.mean(clip_encoded_prompt,     dim=0), dim=0)
        else:
            loss = loss - torch.cosine_similarity(torch.mean(clip_encoded_render, dim=0, keepdim=True), clip_encoded_prompt    )
        loss.backward(retain_graph=True)
    
        # normal loss
        normloss = 0.0
        for i in range(4):
            clip_encoded_render = clip_model.encode_image(norm_augment_transform(rendered_images))
            
            if clip_encoded_prompt.shape[0] > 1:
                normloss = normloss - torch.cosine_similarity(torch.mean(clip_encoded_render, dim=0), torch.mean(clip_encoded_prompt, dim=0), dim=0)    
            else:
                normloss = normloss - torch.cosine_similarity(torch.mean(clip_encoded_render, dim=0, keepdim=True), clip_encoded_prompt)
        normloss.backward(retain_graph=True)

        # colorless loss (geometric loss)
        colorless_loss = 0.0
        default_color = torch.zeros(len(mesh.vertices), 3).to(device)
        default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
        this_mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                        this_mesh.faces) # update the color to grey

        colorless_rendered_images, _, _ = render.render_front_views(this_mesh, 
                                                                    n_views=5,
                                                                    background=background)
        for i in range(4):
           clip_encoded_render = clip_model.encode_image(displacement_augment_transform(colorless_rendered_images))
           
           if clip_encoded_prompt.shape[0] > 1:
               colorless_loss = colorless_loss - torch.cosine_similarity(torch.mean(clip_encoded_render, dim=0), torch.mean(clip_encoded_prompt, dim=0), dim=0)
           else:
               colorless_loss = colorless_loss - torch.cosine_similarity(torch.mean(clip_encoded_render, dim=0, keepdim=True), clip_encoded_prompt)
        colorless_loss.backward(retain_graph=True)

        for param in net.mlp_color.parameters():
            param.requires_grad = True
        for param in net.mlp_norm.parameters():
            param.requires_grad = True

        optim.step()
        lr_scheduler.step()


        this_loss = loss.item()
        if iter % 100 == 0:
            print(f'iter {iter}/{n_iter} || loss={this_loss:.4f}')
            save_path = os.path.join(out_dir, 'iter_{}.jpg'.format(iter))
            torchvision.utils.save_image(rendered_images, save_path)

        optim.zero_grad()
            
    
    # save color and vertices
    with torch.no_grad():
        pred_color, pred_normal = net(net_input)
        pred_normal = pred_normal.detach()
        pred_color = pred_color.detach()
        torch.save(pred_color.cpu(), os.path.join(out_dir, f"pred_color.pt"))
        torch.save(pred_normal.cpu(), os.path.join(out_dir, f"pred_normal.pt"))
        
    # save model
    ckpt = {'iter':n_iter,
            'net':net.state_dict(),
            'optimizer':optim.state_dict()}
    torch.save(ckpt, os.path.join(out_dir, f"final_model.pth"))  