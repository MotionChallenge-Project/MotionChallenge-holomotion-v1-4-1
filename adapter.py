"""SDK-native HoloMotion v1.4.1: author observation schema, recurrent ONNX."""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import ONNXController,yaw


class HoloMotion(ONNXController):
    action_output='actions'

    def __init__(self,root,runtime):
        super().__init__(root,runtime)
        if self.session.get_inputs()[0].shape != [1,self.cfg['input_dimension']]:
            raise ValueError('checkpoint observation dimension differs from author config')

    def reset(self,state,reference,seed):
        super().reset(state,reference,seed)
        self.memory=np.zeros((1,2,1,32,4,64),np.float32);self.step=0
        self.heading=R.from_rotvec([0,0,float(yaw(R.from_quat(state.root_orientation_xyzw))-yaw(R.from_quat(reference[0].root_orientation_xyzw)))])

    def observations(self,state,reference):
        q,qd,omega,gravity,actual=self.proprio(state)
        refq=np.array([f.joint_position_rad for f in reference])[:,self.order]
        positions=np.array([f.root_position_world_m for f in reference])
        rotations=R.from_quat([f.root_orientation_xyzw for f in reference])
        grav=rotations.inv().apply(np.tile([0,0,-1],(11,1)))
        linear=rotations.inv().apply([f.root_linear_velocity_world_m_s for f in reference])
        angular=rotations.inv().apply([f.root_angular_velocity_world_rad_s for f in reference])
        ref_yaw=yaw(rotations)
        heading_error=yaw(self.heading*rotations[0])-yaw(actual)
        yaw_deltas=ref_yaw[1:]-ref_yaw[0]
        relative=(actual.inv()*self.heading*rotations[1:]).as_matrix()
        # First two matrix columns, flattened in row-major order.
        rot6=relative[:,:,:2].reshape(-1)
        current=np.concatenate([grav[0],linear[0],angular[0],refq[0],positions[0,2:3],
            [np.sin(heading_error),np.cos(heading_error)],gravity,omega,q-self.default,qd,self.last])
        future=np.concatenate([refq[1:].reshape(-1),positions[1:,2],grav[1:].reshape(-1),linear[1:].reshape(-1),
            angular[1:].reshape(-1),np.stack([np.sin(yaw_deltas),np.cos(yaw_deltas)],axis=-1).reshape(-1),rot6])
        if self.step>=self.cfg['rope_reset_at']:self.memory.fill(0);self.step=0
        inputs={'obs':np.r_[current,future][None],'past_key_values':self.memory,'step_idx':np.array([self.step],np.int64)}
        self.step+=1
        return inputs

    def after_inference(self,result):self.memory=result['present_key_values']

    def act(self,state,reference):
        command=super().act(state,reference)
        target=self.default.astype(np.float32)+self.scale.astype(np.float32)*self.last
        return replace(command,joint_position_target_rad=tuple(float(x) for x in target[self.inverse]))


def create(root,runtime):return HoloMotion(root,runtime)
