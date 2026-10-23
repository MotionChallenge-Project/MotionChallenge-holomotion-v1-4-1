"""Compare original HoloMotion offline observations, recurrent inference and PD targets.

uv run validate_inference.py --source /path/to/pinned/HoloMotion --weights MODEL --output REPORT
Only ROS/import glue is omitted from the original source AST. No training simulator is required.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
from adapter import HoloMotion
from motion_challenge.sdk import StatePacket, ReferenceFrame


def definitions(path, scope, names=None, methods=None):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
             and (names is None or n.name in names)]
    if methods is not None:
        for node in nodes:
            node.body = [m for m in node.body if isinstance(m,ast.FunctionDef) and m.name in methods]
    exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+nodes,
                           type_ignores=[]),str(path),'exec'),scope)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    pin=json.loads((root/'native-sources.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,expected in pin['files'].items():
        assert sha(args.source/name)==expected, name
    assert sha(root/'validation/author-config.yaml')==pin['checkpoint_config_sha256']
    config=yaml.safe_load((root/'validation/author-config.yaml').read_text())
    runtime=json.loads((root/'runtime.json').read_text());runtime['weights']=str(args.weights.resolve())
    policy=HoloMotion(root,runtime)
    metadata=policy.session.get_modelmeta().custom_metadata_map
    assert metadata==pin['onnx_metadata']
    assert policy.names==metadata['joint_names'].split(',')
    for attr,key in [('default','default_joint_pos'),('kp','joint_stiffness'),('kd','joint_damping'),('scale','action_scale')]:
        expected=np.array([float(v) for v in metadata[key].split(',')],np.float32)
        np.testing.assert_array_equal(getattr(policy,attr),expected)
    assert not config['robot']['actuators'].get('ema_filter_enabled',False)
    native=args.source/'deployment/unitree_g1_ros2_29dof/src/humanoid_policy'
    scope={'__name__':__name__,'np':np,'dataclass':dataclass,'Sequence':Sequence}
    definitions(native/'offline_motion_reference.py',scope)
    definitions(native/'observation_evaluator.py',scope)
    methods={'reset_counters','_maybe_reset_motion_rope_window','_run_current_policy','apply_policy_output'}
    definitions(native/'policy_runtime.py',scope,['PolicyRuntime'],methods)
    schema=config['modules']['actor']['obs_schema']
    terms=[name.split('/')[-1] for key in ('flattened_obs','flattened_obs_fut') for name in schema[key]['terms']]
    cfg_groups=config['obs']['obs_groups']['unified']
    atomic={k:v for group in ('atomic_obs_list','additional_atomic_obs_list') for item in cfg_groups.get(group,[]) for k,v in item.items()}
    assert all(atomic[t].get('scale',1)==1 for t in terms)
    assert config['obs']['context_length']==1 and config['obs']['n_fut_frames']==10 and config['obs']['target_fps']==50
    rng=np.random.default_rng(271023)
    names=tuple(np.asarray(policy.names)[rng.permutation(29)])
    log=NS(info=lambda *_:None,warn=lambda *_:None)
    errors=dict(observation=0.,network=0.,action=0.,motor_target=0.,cache=0.,reset_cache=0.)
    def error(name,a,b):
        errors[name]=max(errors[name],float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    samples=0
    for reset in range(4):
        node=NS(dof_names_ref_motion=list(names),motion_dof_names_onnx=policy.names,velocity_dof_names_onnx=policy.names,
                motion_default_angles_onnx=policy.default.astype(np.float32),velocity_default_angles_onnx=policy.default.astype(np.float32),
                real_dof_names=list(names),actions_dim=29,num_actions=29,n_fut_frames=10,current_policy_mode='motion',
                motion_frame_idx=0,reference_stream_active=False,motion_uses_vr_reference=False,latest_obs_flag=False,_vr_reference=None,
                _lowstate_msg=None,actions_onnx=np.zeros(29,np.float32),get_logger=lambda:log)
        observer=scope['PolicyObservationEvaluator'](node);observer.initialize_observation_state()
        port=NS(use_kv_cache=True,motion_kv_cache=np.zeros((1,2,1,32,4,64),np.float32),
                motion_kv_shape=[1,2,1,32,4,64],motion_kv_dtype=np.float32,
                motion_input_name='obs',motion_output_name='actions',motion_kv_input_name='past_key_values',
                motion_kv_output_name='present_key_values',motion_step_idx_input_name='step_idx',
                motion_policy_session=policy.session,motion_rope_max_seq_len=int(metadata.get('rope_max_seq_len',0)),
                motion_rope_reset_margin=64,motion_action_ema_filter_enabled=False,get_logger=lambda:log)
        native_policy=scope['PolicyRuntime']();native_policy.port=port
        native_policy.state=NS(current_policy_mode='motion',motion_step_idx=0,motion_frame_idx=0,
            actions_onnx=np.zeros(29,np.float32),target_dof_pos_onnx=np.zeros(29,np.float32),
            target_dof_pos_real_buffers=[np.zeros(29,np.float32),np.zeros(29,np.float32)],target_dof_pos_real_buffer_idx=0)
        native_policy._motion_input_feed=None;native_policy._motion_output_names=None;native_policy._motion_step_idx_array=None
        native_policy.reset_counters()
        for step in range(40):
            actual=R.random(random_state=rng)
            q=rng.normal(0,.3,29);qd=rng.normal(0,.5,29);omega=rng.normal(0,.3,3)
            state=StatePacket(step*.02,names,tuple(q),tuple(qd),(0,0,.8),tuple(actual.as_quat()),
                              (0,0,0),tuple(actual.apply(omega)))
            refs=[]
            for i in range(11):
                ref_rot=R.random(random_state=rng)
                refs.append(ReferenceFrame(step*.02+i*.02,step*.02+i*.02,tuple(rng.normal(0,.3,29)),
                    tuple(rng.normal(0,.3,3)),tuple(ref_rot.as_quat()),tuple(rng.normal(0,.5,29)),
                    tuple(rng.normal(0,.5,3)),tuple(rng.normal(0,.3,3))))
            if step>=35:refs[3:]=[refs[2]]*8
            refs=tuple(refs)
            clip=NS(n_frames=11,dof_pos=np.asarray([r.joint_position_rad for r in refs],np.float32),
                dof_vel=np.asarray([r.joint_velocity_rad_s for r in refs],np.float32),
                global_translation=np.asarray([r.root_position_world_m for r in refs],np.float32)[:,None],
                global_rotation_quat=np.asarray([r.root_orientation_xyzw for r in refs],np.float32)[:,None],
                global_velocity=np.asarray([r.root_linear_velocity_world_m_s for r in refs],np.float32)[:,None],
                global_angular_velocity=np.asarray([r.root_angular_velocity_world_rad_s for r in refs],np.float32)[:,None])
            observer._offline_reference.set_clip(clip)
            node._lowstate_msg=NS(imu_state=NS(quaternion=actual.as_quat()[[3,0,1,2]],gyroscope=omega),
                                  motor_state=[NS(q=a,dq=b) for a,b in zip(q,qd)])
            node.actions_onnx=native_policy.state.actions_onnx
            if step==0:
                policy.reset(state,refs,reset)
                if hasattr(observer,'begin_motion_yaw_alignment'):observer.begin_motion_yaw_alignment()
                error('reset_cache',policy.memory,port.motion_kv_cache)
            if step==37 and policy.cfg['rope_reset_at'] is not None:
                policy.step=native_policy.state.motion_step_idx=policy.cfg['rope_reset_at']-1
            native_obs=np.concatenate([np.asarray(getattr(observer,'_get_obs_'+term)()).reshape(-1).copy() for term in terms])[None].astype(np.float32)
            actual_inputs=policy.observations(state,refs)
            error('observation',actual_inputs['obs'],native_obs)
            expected=native_policy._run_current_policy(native_obs)
            identical=policy.infer(native_policy._motion_input_feed)
            error('network',identical['actions'],expected[0])
            targets=native_policy.apply_policy_output(expected[0],action_scale=policy.scale.astype(np.float32),
                default_angles=policy.default.astype(np.float32),onnx_to_real=policy.inverse,is_motion=True)
            observation_method=policy.observations;policy.observations=lambda *_:actual_inputs
            command=policy.act(state,refs);policy.observations=observation_method
            error('action',policy.last,native_policy.state.actions_onnx)
            error('motor_target',command.joint_position_target_rad,targets)
            error('cache',policy.memory,port.motion_kv_cache)
            assert policy.step==native_policy.state.motion_step_idx
            samples+=1
    limits=dict(observation=2e-4,network=0.,action=2e-3,motor_target=2e-3,cache=2e-3,reset_cache=0.)
    report={'passed':all(errors[k]<=limits[k] for k in errors),'cases':samples,'resets':4,'max_abs_error':errors,'limits':limits,
            'checkpoint_sha256':runtime['weights_sha256'],'source_commit':pin['source_commit'],'version':pin['version'],
            'scope':'Original offline-reference/evaluator/runtime AST, published ONNX metadata, recurrent ONNX and motor targets; no ROS glue. Shared benchmark physics and GPU-provider equivalence are separate.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':main()
