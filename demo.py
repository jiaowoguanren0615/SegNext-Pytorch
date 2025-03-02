


# 定义两组 keys
keys_set1 = {'backbone.patch_embed1.proj.0.weight', 'backbone.patch_embed1.proj.0.bias',
             'backbone.patch_embed1.proj.1.weight', 'backbone.patch_embed1.proj.1.bias',
             'backbone.patch_embed1.proj.1.running_mean', 'backbone.patch_embed1.proj.1.running_var',
             'backbone.patch_embed1.proj.1.num_batches_tracked', 'backbone.patch_embed1.proj.3.weight',
             'backbone.patch_embed1.proj.3.bias', 'backbone.patch_embed1.proj.4.weight',
             'backbone.patch_embed1.proj.4.bias', 'backbone.patch_embed1.proj.4.running_mean',
             'backbone.patch_embed1.proj.4.running_var', 'backbone.patch_embed1.proj.4.num_batches_tracked',
             'backbone.block1.0.layer_scale_1', 'backbone.block1.0.layer_scale_2', 'backbone.block1.0.norm1.weight',
             'backbone.block1.0.norm1.bias', 'backbone.block1.0.norm1.running_mean',
             'backbone.block1.0.norm1.running_var', 'backbone.block1.0.norm1.num_batches_tracked',
             'backbone.block1.0.attn.proj_1.weight', 'backbone.block1.0.attn.proj_1.bias',
             'backbone.block1.0.attn.spatial_gating_unit.conv0.weight',
             'backbone.block1.0.attn.spatial_gating_unit.conv0.bias',
             'backbone.block1.0.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block1.0.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block1.0.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block1.0.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block1.0.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block1.0.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block1.0.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block1.0.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block1.0.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block1.0.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block1.0.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block1.0.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block1.0.attn.spatial_gating_unit.conv3.weight',
             'backbone.block1.0.attn.spatial_gating_unit.conv3.bias', 'backbone.block1.0.attn.proj_2.weight',
             'backbone.block1.0.attn.proj_2.bias', 'backbone.block1.0.norm2.weight', 'backbone.block1.0.norm2.bias',
             'backbone.block1.0.norm2.running_mean', 'backbone.block1.0.norm2.running_var',
             'backbone.block1.0.norm2.num_batches_tracked', 'backbone.block1.0.mlp.fc1.weight',
             'backbone.block1.0.mlp.fc1.bias', 'backbone.block1.0.mlp.dwconv.dwconv.weight',
             'backbone.block1.0.mlp.dwconv.dwconv.bias', 'backbone.block1.0.mlp.fc2.weight',
             'backbone.block1.0.mlp.fc2.bias', 'backbone.block1.1.layer_scale_1', 'backbone.block1.1.layer_scale_2',
             'backbone.block1.1.norm1.weight', 'backbone.block1.1.norm1.bias', 'backbone.block1.1.norm1.running_mean',
             'backbone.block1.1.norm1.running_var', 'backbone.block1.1.norm1.num_batches_tracked',
             'backbone.block1.1.attn.proj_1.weight', 'backbone.block1.1.attn.proj_1.bias',
             'backbone.block1.1.attn.spatial_gating_unit.conv0.weight',
             'backbone.block1.1.attn.spatial_gating_unit.conv0.bias',
             'backbone.block1.1.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block1.1.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block1.1.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block1.1.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block1.1.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block1.1.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block1.1.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block1.1.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block1.1.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block1.1.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block1.1.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block1.1.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block1.1.attn.spatial_gating_unit.conv3.weight',
             'backbone.block1.1.attn.spatial_gating_unit.conv3.bias', 'backbone.block1.1.attn.proj_2.weight',
             'backbone.block1.1.attn.proj_2.bias', 'backbone.block1.1.norm2.weight', 'backbone.block1.1.norm2.bias',
             'backbone.block1.1.norm2.running_mean', 'backbone.block1.1.norm2.running_var',
             'backbone.block1.1.norm2.num_batches_tracked', 'backbone.block1.1.mlp.fc1.weight',
             'backbone.block1.1.mlp.fc1.bias', 'backbone.block1.1.mlp.dwconv.dwconv.weight',
             'backbone.block1.1.mlp.dwconv.dwconv.bias', 'backbone.block1.1.mlp.fc2.weight',
             'backbone.block1.1.mlp.fc2.bias', 'backbone.norm1.weight', 'backbone.norm1.bias',
             'backbone.patch_embed2.proj.weight', 'backbone.patch_embed2.proj.bias',
             'backbone.patch_embed2.norm.weight', 'backbone.patch_embed2.norm.bias',
             'backbone.patch_embed2.norm.running_mean', 'backbone.patch_embed2.norm.running_var',
             'backbone.patch_embed2.norm.num_batches_tracked', 'backbone.block2.0.layer_scale_1',
             'backbone.block2.0.layer_scale_2', 'backbone.block2.0.norm1.weight', 'backbone.block2.0.norm1.bias',
             'backbone.block2.0.norm1.running_mean', 'backbone.block2.0.norm1.running_var',
             'backbone.block2.0.norm1.num_batches_tracked', 'backbone.block2.0.attn.proj_1.weight',
             'backbone.block2.0.attn.proj_1.bias', 'backbone.block2.0.attn.spatial_gating_unit.conv0.weight',
             'backbone.block2.0.attn.spatial_gating_unit.conv0.bias',
             'backbone.block2.0.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block2.0.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block2.0.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block2.0.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block2.0.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block2.0.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block2.0.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block2.0.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block2.0.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block2.0.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block2.0.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block2.0.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block2.0.attn.spatial_gating_unit.conv3.weight',
             'backbone.block2.0.attn.spatial_gating_unit.conv3.bias', 'backbone.block2.0.attn.proj_2.weight',
             'backbone.block2.0.attn.proj_2.bias', 'backbone.block2.0.norm2.weight', 'backbone.block2.0.norm2.bias',
             'backbone.block2.0.norm2.running_mean', 'backbone.block2.0.norm2.running_var',
             'backbone.block2.0.norm2.num_batches_tracked', 'backbone.block2.0.mlp.fc1.weight',
             'backbone.block2.0.mlp.fc1.bias', 'backbone.block2.0.mlp.dwconv.dwconv.weight',
             'backbone.block2.0.mlp.dwconv.dwconv.bias', 'backbone.block2.0.mlp.fc2.weight',
             'backbone.block2.0.mlp.fc2.bias', 'backbone.block2.1.layer_scale_1', 'backbone.block2.1.layer_scale_2',
             'backbone.block2.1.norm1.weight', 'backbone.block2.1.norm1.bias', 'backbone.block2.1.norm1.running_mean',
             'backbone.block2.1.norm1.running_var', 'backbone.block2.1.norm1.num_batches_tracked',
             'backbone.block2.1.attn.proj_1.weight', 'backbone.block2.1.attn.proj_1.bias',
             'backbone.block2.1.attn.spatial_gating_unit.conv0.weight',
             'backbone.block2.1.attn.spatial_gating_unit.conv0.bias',
             'backbone.block2.1.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block2.1.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block2.1.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block2.1.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block2.1.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block2.1.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block2.1.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block2.1.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block2.1.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block2.1.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block2.1.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block2.1.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block2.1.attn.spatial_gating_unit.conv3.weight',
             'backbone.block2.1.attn.spatial_gating_unit.conv3.bias', 'backbone.block2.1.attn.proj_2.weight',
             'backbone.block2.1.attn.proj_2.bias', 'backbone.block2.1.norm2.weight', 'backbone.block2.1.norm2.bias',
             'backbone.block2.1.norm2.running_mean', 'backbone.block2.1.norm2.running_var',
             'backbone.block2.1.norm2.num_batches_tracked', 'backbone.block2.1.mlp.fc1.weight',
             'backbone.block2.1.mlp.fc1.bias', 'backbone.block2.1.mlp.dwconv.dwconv.weight',
             'backbone.block2.1.mlp.dwconv.dwconv.bias', 'backbone.block2.1.mlp.fc2.weight',
             'backbone.block2.1.mlp.fc2.bias', 'backbone.norm2.weight', 'backbone.norm2.bias',
             'backbone.patch_embed3.proj.weight', 'backbone.patch_embed3.proj.bias',
             'backbone.patch_embed3.norm.weight', 'backbone.patch_embed3.norm.bias',
             'backbone.patch_embed3.norm.running_mean', 'backbone.patch_embed3.norm.running_var',
             'backbone.patch_embed3.norm.num_batches_tracked', 'backbone.block3.0.layer_scale_1',
             'backbone.block3.0.layer_scale_2', 'backbone.block3.0.norm1.weight', 'backbone.block3.0.norm1.bias',
             'backbone.block3.0.norm1.running_mean', 'backbone.block3.0.norm1.running_var',
             'backbone.block3.0.norm1.num_batches_tracked', 'backbone.block3.0.attn.proj_1.weight',
             'backbone.block3.0.attn.proj_1.bias', 'backbone.block3.0.attn.spatial_gating_unit.conv0.weight',
             'backbone.block3.0.attn.spatial_gating_unit.conv0.bias',
             'backbone.block3.0.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block3.0.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block3.0.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block3.0.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block3.0.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block3.0.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block3.0.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block3.0.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block3.0.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block3.0.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block3.0.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block3.0.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block3.0.attn.spatial_gating_unit.conv3.weight',
             'backbone.block3.0.attn.spatial_gating_unit.conv3.bias', 'backbone.block3.0.attn.proj_2.weight',
             'backbone.block3.0.attn.proj_2.bias', 'backbone.block3.0.norm2.weight', 'backbone.block3.0.norm2.bias',
             'backbone.block3.0.norm2.running_mean', 'backbone.block3.0.norm2.running_var',
             'backbone.block3.0.norm2.num_batches_tracked', 'backbone.block3.0.mlp.fc1.weight',
             'backbone.block3.0.mlp.fc1.bias', 'backbone.block3.0.mlp.dwconv.dwconv.weight',
             'backbone.block3.0.mlp.dwconv.dwconv.bias', 'backbone.block3.0.mlp.fc2.weight',
             'backbone.block3.0.mlp.fc2.bias', 'backbone.block3.1.layer_scale_1', 'backbone.block3.1.layer_scale_2',
             'backbone.block3.1.norm1.weight', 'backbone.block3.1.norm1.bias', 'backbone.block3.1.norm1.running_mean',
             'backbone.block3.1.norm1.running_var', 'backbone.block3.1.norm1.num_batches_tracked',
             'backbone.block3.1.attn.proj_1.weight', 'backbone.block3.1.attn.proj_1.bias',
             'backbone.block3.1.attn.spatial_gating_unit.conv0.weight',
             'backbone.block3.1.attn.spatial_gating_unit.conv0.bias',
             'backbone.block3.1.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block3.1.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block3.1.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block3.1.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block3.1.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block3.1.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block3.1.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block3.1.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block3.1.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block3.1.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block3.1.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block3.1.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block3.1.attn.spatial_gating_unit.conv3.weight',
             'backbone.block3.1.attn.spatial_gating_unit.conv3.bias', 'backbone.block3.1.attn.proj_2.weight',
             'backbone.block3.1.attn.proj_2.bias', 'backbone.block3.1.norm2.weight', 'backbone.block3.1.norm2.bias',
             'backbone.block3.1.norm2.running_mean', 'backbone.block3.1.norm2.running_var',
             'backbone.block3.1.norm2.num_batches_tracked', 'backbone.block3.1.mlp.fc1.weight',
             'backbone.block3.1.mlp.fc1.bias', 'backbone.block3.1.mlp.dwconv.dwconv.weight',
             'backbone.block3.1.mlp.dwconv.dwconv.bias', 'backbone.block3.1.mlp.fc2.weight',
             'backbone.block3.1.mlp.fc2.bias', 'backbone.block3.2.layer_scale_1', 'backbone.block3.2.layer_scale_2',
             'backbone.block3.2.norm1.weight', 'backbone.block3.2.norm1.bias', 'backbone.block3.2.norm1.running_mean',
             'backbone.block3.2.norm1.running_var', 'backbone.block3.2.norm1.num_batches_tracked',
             'backbone.block3.2.attn.proj_1.weight', 'backbone.block3.2.attn.proj_1.bias',
             'backbone.block3.2.attn.spatial_gating_unit.conv0.weight',
             'backbone.block3.2.attn.spatial_gating_unit.conv0.bias',
             'backbone.block3.2.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block3.2.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block3.2.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block3.2.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block3.2.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block3.2.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block3.2.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block3.2.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block3.2.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block3.2.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block3.2.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block3.2.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block3.2.attn.spatial_gating_unit.conv3.weight',
             'backbone.block3.2.attn.spatial_gating_unit.conv3.bias', 'backbone.block3.2.attn.proj_2.weight',
             'backbone.block3.2.attn.proj_2.bias', 'backbone.block3.2.norm2.weight', 'backbone.block3.2.norm2.bias',
             'backbone.block3.2.norm2.running_mean', 'backbone.block3.2.norm2.running_var',
             'backbone.block3.2.norm2.num_batches_tracked', 'backbone.block3.2.mlp.fc1.weight',
             'backbone.block3.2.mlp.fc1.bias', 'backbone.block3.2.mlp.dwconv.dwconv.weight',
             'backbone.block3.2.mlp.dwconv.dwconv.bias', 'backbone.block3.2.mlp.fc2.weight',
             'backbone.block3.2.mlp.fc2.bias', 'backbone.block3.3.layer_scale_1', 'backbone.block3.3.layer_scale_2',
             'backbone.block3.3.norm1.weight', 'backbone.block3.3.norm1.bias', 'backbone.block3.3.norm1.running_mean',
             'backbone.block3.3.norm1.running_var', 'backbone.block3.3.norm1.num_batches_tracked',
             'backbone.block3.3.attn.proj_1.weight', 'backbone.block3.3.attn.proj_1.bias',
             'backbone.block3.3.attn.spatial_gating_unit.conv0.weight',
             'backbone.block3.3.attn.spatial_gating_unit.conv0.bias',
             'backbone.block3.3.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block3.3.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block3.3.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block3.3.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block3.3.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block3.3.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block3.3.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block3.3.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block3.3.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block3.3.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block3.3.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block3.3.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block3.3.attn.spatial_gating_unit.conv3.weight',
             'backbone.block3.3.attn.spatial_gating_unit.conv3.bias', 'backbone.block3.3.attn.proj_2.weight',
             'backbone.block3.3.attn.proj_2.bias', 'backbone.block3.3.norm2.weight', 'backbone.block3.3.norm2.bias',
             'backbone.block3.3.norm2.running_mean', 'backbone.block3.3.norm2.running_var',
             'backbone.block3.3.norm2.num_batches_tracked', 'backbone.block3.3.mlp.fc1.weight',
             'backbone.block3.3.mlp.fc1.bias', 'backbone.block3.3.mlp.dwconv.dwconv.weight',
             'backbone.block3.3.mlp.dwconv.dwconv.bias', 'backbone.block3.3.mlp.fc2.weight',
             'backbone.block3.3.mlp.fc2.bias', 'backbone.norm3.weight', 'backbone.norm3.bias',
             'backbone.patch_embed4.proj.weight', 'backbone.patch_embed4.proj.bias',
             'backbone.patch_embed4.norm.weight', 'backbone.patch_embed4.norm.bias',
             'backbone.patch_embed4.norm.running_mean', 'backbone.patch_embed4.norm.running_var',
             'backbone.patch_embed4.norm.num_batches_tracked', 'backbone.block4.0.layer_scale_1',
             'backbone.block4.0.layer_scale_2', 'backbone.block4.0.norm1.weight', 'backbone.block4.0.norm1.bias',
             'backbone.block4.0.norm1.running_mean', 'backbone.block4.0.norm1.running_var',
             'backbone.block4.0.norm1.num_batches_tracked', 'backbone.block4.0.attn.proj_1.weight',
             'backbone.block4.0.attn.proj_1.bias', 'backbone.block4.0.attn.spatial_gating_unit.conv0.weight',
             'backbone.block4.0.attn.spatial_gating_unit.conv0.bias',
             'backbone.block4.0.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block4.0.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block4.0.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block4.0.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block4.0.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block4.0.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block4.0.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block4.0.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block4.0.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block4.0.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block4.0.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block4.0.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block4.0.attn.spatial_gating_unit.conv3.weight',
             'backbone.block4.0.attn.spatial_gating_unit.conv3.bias', 'backbone.block4.0.attn.proj_2.weight',
             'backbone.block4.0.attn.proj_2.bias', 'backbone.block4.0.norm2.weight', 'backbone.block4.0.norm2.bias',
             'backbone.block4.0.norm2.running_mean', 'backbone.block4.0.norm2.running_var',
             'backbone.block4.0.norm2.num_batches_tracked', 'backbone.block4.0.mlp.fc1.weight',
             'backbone.block4.0.mlp.fc1.bias', 'backbone.block4.0.mlp.dwconv.dwconv.weight',
             'backbone.block4.0.mlp.dwconv.dwconv.bias', 'backbone.block4.0.mlp.fc2.weight',
             'backbone.block4.0.mlp.fc2.bias', 'backbone.block4.1.layer_scale_1', 'backbone.block4.1.layer_scale_2',
             'backbone.block4.1.norm1.weight', 'backbone.block4.1.norm1.bias', 'backbone.block4.1.norm1.running_mean',
             'backbone.block4.1.norm1.running_var', 'backbone.block4.1.norm1.num_batches_tracked',
             'backbone.block4.1.attn.proj_1.weight', 'backbone.block4.1.attn.proj_1.bias',
             'backbone.block4.1.attn.spatial_gating_unit.conv0.weight',
             'backbone.block4.1.attn.spatial_gating_unit.conv0.bias',
             'backbone.block4.1.attn.spatial_gating_unit.conv0_1.weight',
             'backbone.block4.1.attn.spatial_gating_unit.conv0_1.bias',
             'backbone.block4.1.attn.spatial_gating_unit.conv0_2.weight',
             'backbone.block4.1.attn.spatial_gating_unit.conv0_2.bias',
             'backbone.block4.1.attn.spatial_gating_unit.conv1_1.weight',
             'backbone.block4.1.attn.spatial_gating_unit.conv1_1.bias',
             'backbone.block4.1.attn.spatial_gating_unit.conv1_2.weight',
             'backbone.block4.1.attn.spatial_gating_unit.conv1_2.bias',
             'backbone.block4.1.attn.spatial_gating_unit.conv2_1.weight',
             'backbone.block4.1.attn.spatial_gating_unit.conv2_1.bias',
             'backbone.block4.1.attn.spatial_gating_unit.conv2_2.weight',
             'backbone.block4.1.attn.spatial_gating_unit.conv2_2.bias',
             'backbone.block4.1.attn.spatial_gating_unit.conv3.weight',
             'backbone.block4.1.attn.spatial_gating_unit.conv3.bias', 'backbone.block4.1.attn.proj_2.weight',
             'backbone.block4.1.attn.proj_2.bias', 'backbone.block4.1.norm2.weight', 'backbone.block4.1.norm2.bias',
             'backbone.block4.1.norm2.running_mean', 'backbone.block4.1.norm2.running_var',
             'backbone.block4.1.norm2.num_batches_tracked', 'backbone.block4.1.mlp.fc1.weight',
             'backbone.block4.1.mlp.fc1.bias', 'backbone.block4.1.mlp.dwconv.dwconv.weight',
             'backbone.block4.1.mlp.dwconv.dwconv.bias', 'backbone.block4.1.mlp.fc2.weight',
             'backbone.block4.1.mlp.fc2.bias'}

keys_set2 = set()

# 计算差集和交集
only_in_set1 = keys_set1 - keys_set2  # 仅在第一组
only_in_set2 = keys_set2 - keys_set1  # 仅在第二组
in_both_sets = keys_set1 & keys_set2  # 两组都有

# 打印结果
print("仅在第一组中的 keys:")
for key in sorted(only_in_set1):
    print(key)

print("\n仅在第二组中的 keys:")
for key in sorted(only_in_set2):
    print(key)

print("\n两组都有的 keys:")
for key in sorted(in_both_sets):
    print(key)





