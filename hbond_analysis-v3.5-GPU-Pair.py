    #!/usr/bin/env python3
"""
基于模板的交互式优化氢键分析工具

该工具允许用户指定构型和轨迹文件路径，自动识别氢键供体和受体，
进行预分析之后给出选择原子类型参考分析那些部分的氢键统计，
并把氢键关联函数作为单独的输出选项。

优化特性：
1. 支持多进程并行计算
2. 支持GPU加速（通过CuPy）
3. 向量化计算优化
4. 支持Numba JIT编译加速（如果可用）
"""

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import time
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

# 尝试导入CuPy用于GPU加速
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    print("CUDA可用，将使用GPU加速计算")
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA不可用，将使用CPU计算")

# 尝试导入Numba用于JIT编译加速
try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("Numba可用，将使用JIT编译加速")
    
    @jit(nopython=True)
    def fast_distance_calc(pos1, pos2):
        """使用Numba加速的距离计算"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        return np.sqrt(dx*dx + dy*dy + dz*dz)
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba不可用")

def get_user_input():
    """
    获取用户输入的文件路径和其他参数
    """
    print("=" * 60)
    print("基于模板的交互式优化氢键分析工具")
    print("=" * 60)
    
    # 获取文件路径
    print("\n请输入文件路径信息:")
    topology_file = input("拓扑文件路径 (如 GO1.data): ").strip()
    trajectory_file = input("轨迹文件路径 (如 moveGO1.lammpstrj): ").strip()
    
    # 检查文件是否存在
    if not os.path.exists(topology_file):
        raise FileNotFoundError(f"拓扑文件不存在: {topology_file}")
    
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"轨迹文件不存在: {trajectory_file}")
    # 获取时间间隔参数
    print("\n分析参数设置:")
    timestep = input("轨迹间隔时间 (默认 1 ps): ").strip()
    timestep = float(timestep) if timestep else 1.0

    # 获取分析参数
    print("\n分析参数设置:")
    distance_cutoff = input("氢键距离阈值 (默认 3.5 Å): ").strip()
    distance_cutoff = float(distance_cutoff) if distance_cutoff else 3.5
    
    angle_cutoff = input("氢键角度阈值 (默认 120.0°): ").strip()
    angle_cutoff = float(angle_cutoff) if angle_cutoff else 120.0
    
    return topology_file, trajectory_file, timestep, distance_cutoff, angle_cutoff

def analyze_system_composition(u):
    """
    分析系统组成并自动识别氢键相关原子
    """
    print("\n" + "-" * 50)
    print("系统组成分析")
    print("-" * 50)
    print(f"系统信息: {u.atoms.n_atoms} 原子")
    print(f"轨迹帧数: {u.trajectory.n_frames}")
    
    # 分析原子类型
    type_counts = Counter([atom.type for atom in u.atoms])
    
    print("\n原子类型统计:")
    for atom_type, count in sorted(type_counts.items()):
        print(f"  类型 {atom_type}: {count:6d} 个原子")
    
    return type_counts

def auto_identify_hbond_atoms(u, type_labels=None):
    """
    自动识别氢键相关原子（供体、氢、受体）
    使用type_labels信息提高识别准确性
    """
    print("\n" + "-" * 50)
    print("自动识别氢键相关原子")
    print("-" * 50)
    
    # 尝试基于原子类型识别，避免使用名称（某些格式不支持）
    # 氢原子识别
    try:
        h_atoms = u.select_atoms("name H* ")
    except:
        # 如果名称选择失败，使用type_labels信息或质量判据选择
        if type_labels:
            # 从type_labels中找出氢原子类型
            h_types = [t for t, l in type_labels.items() if l.startswith('H')]
            if h_types:
                h_atoms = u.select_atoms(f"type {' '.join(map(str, h_types))}")
            else:
                # 如果没有找到明确的氢原子类型，使用质量判据
                h_atoms = u.select_atoms("mass 0.5:1.5")
        else:
            # 如果没有type_labels信息，使用质量判据
            h_atoms = u.select_atoms("mass 0.5:1.5")
    
    # 氧/氮/氟等可能的供体/受体原子 (不包括硫等其他较重元素，以提高精确度)
    try:
        o_atoms = u.select_atoms("name O* N* F*")
    except:
        # 基于type_labels或质量识别重原子 (O, N, F元素的质量范围更精确)
        if type_labels:
            # 从type_labels中找出氧、氮、氟原子类型
            onf_types = []
            for t, l in type_labels.items():
                # 检查标签是否以O, N, F开头（考虑下划线等情况）
                if any(l.startswith(prefix) for prefix in ['O', 'N', 'F']):
                    onf_types.append(t)
            
            if onf_types:
                o_atoms = u.select_atoms(f"type {' '.join(map(str, onf_types))}")
            else:
                # 如果没有找到明确的O/N/F类型，使用质量范围
                o_atoms = u.select_atoms("mass 13.0:20.0")
        else:
            # 如果没有type_labels信息，使用质量范围
            o_atoms = u.select_atoms("mass 13.0:20.0")
    
    print(f"自动识别结果:")
    print(f"  氢原子: {len(h_atoms):6d} 个")
    print(f"  供体/受体原子: {len(o_atoms):6d} 个")
    
    # 显示详细的识别结果信息，方便用户自查
    if len(h_atoms) > 0:
        # 统计氢原子的类型
        h_types = list(set([atom.type for atom in h_atoms]))
        print(f"    氢原子类型: {', '.join(map(str, h_types))}")
        
        # 如果有type_labels信息，显示对应的标签
        if type_labels and len(type_labels) > 0:
            h_labels = list(set([type_labels.get(atom.type, '') for atom in h_atoms]))
            print(f"    氢原子标签: {', '.join(h_labels)}")
        
        # 检查是否有名称信息
        try:
            h_names = list(set([atom.name for atom in h_atoms]))
            print(f"    氢原子名称: {', '.join(h_names)}")
        except:
            print("    氢原子名称: 无法获取（轨迹文件中未包含名称信息）")
    
    if len(o_atoms) > 0:
        # 统计供体/受体原子的类型
        o_types = list(set([atom.type for atom in o_atoms]))
        print(f"    供体/受体原子类型: {', '.join(map(str, o_types))}")
        
        # 如果有type_labels信息，显示对应的标签
        if type_labels and len(type_labels) > 0:
            o_labels = list(set([type_labels.get(atom.type, '') for atom in o_atoms]))
            print(f"    供体/受体原子标签: {', '.join(o_labels)}")
        
        # 检查是否有名称信息
        try:
            o_names = list(set([atom.name for atom in o_atoms]))
            print(f"    供体/受体原子名称: {', '.join(o_names)}")
        except:
            print("    供体/受体原子名称: 无法获取（轨迹文件中未包含名称信息）")
    
    return h_atoms, o_atoms

def select_atom_types(u, type_labels=None):
    """
    让用户选择要分析的原子类型
    """
    print("\n" + "-" * 50)
    print("原子类型选择")
    print("-" * 50)
    
    # 显示Atom Type Labels信息（如果可用）
    if type_labels:
        print("检测到Atom Type Labels信息:")
        for atom_type, label in sorted(type_labels.items()):
            print(f"  类型 {atom_type}: {label}")
        print()
    
    print("1. 使用自动识别的原子类型")
    print("2. 手动指定原子类型")
    
    choice = input("请选择 (默认 1): ").strip() or "1"
    
    if choice == "1":
        h_atoms, o_atoms = auto_identify_hbond_atoms(u, type_labels)
    else:
        print("\n手动指定原子类型:")
        print("支持的语法:")
        print("  - 按类型选择: type 1 2")
        print("  - 按质量选择: mass 0.5:1.5")
        if type_labels:
            print("  - 根据Atom Type Labels，可用的类型标签:")
            for atom_type, label in sorted(type_labels.items()):
                print(f"    {label} (类型 {atom_type})")
        print()
        print("常见氢键元素的类型选择参考: O N F S P ")
        print(" 可以组合多种类型，如: type 1 2 5 8 9")
        print()
        
        h_selection = input("氢原子选择语句 (如 'type 11 13'): ").strip()
        o_selection = input("供体/受体原子选择语句 (如 'type 1 8 9 10 12'): ").strip()
        
        try:
            h_atoms = u.select_atoms(h_selection) if h_selection else None
            o_atoms = u.select_atoms(o_selection) if o_selection else None
            
            if h_atoms is not None and o_atoms is not None:
                print(f"\n手动选择结果:")
                print(f"  氢原子: {len(h_atoms):6d} 个")
                print(f"  供体/受体原子: {len(o_atoms):6d} 个")
            else:
                print(f"选择语句有误")
                raise ValueError("无效的选择语句")
        except Exception as e:
            print(f"选择语句有误: {e}")
            print("尝试使用默认质量判据选择...")
            try:
                # 根据项目规范，使用范围选择语法代替比较运算符
                h_atoms = u.select_atoms("mass 0.5:1.5")
                o_atoms = u.select_atoms("mass 10.0:20.0")
                print("使用默认质量判据选择成功!")
                print(f"\n自动选择结果:")
                print(f"  氢原子: {len(h_atoms):6d} 个")
                print(f"  供体/受体原子: {len(o_atoms):6d} 个")
            except:
                print("自动选择失败，请检查类型选择")
                    
    return h_atoms, o_atoms

def precompute_donor_info(h_atoms, o_atoms):
    """
    预计算氢原子和供体原子的信息，提高分析效率
    """
    print("\n预计算氢键相关信息...")
    precompute_start = time.time()
    
    # 为每个氢原子找到可能的供体氧原子
    h_donors = {}  # 存储每个氢原子最可能的供体
    o_positions = o_atoms.positions
    
    for h in h_atoms:
        h_id = h.id
        h_pos = h.position
        h_donors[h_id] = []
        
        # 计算与所有氧原子的距离
        for o in o_atoms:
            o_pos = o.position
            dist = np.linalg.norm(h_pos - o_pos)
            
            # 如果在共价键范围内，可能是供体
            if 0.8 <= dist <= 1.2:
                h_donors[h_id].append(o.id)
    
    precompute_time = time.time() - precompute_start
    print(f"预计算完成，用时: {precompute_time:.2f} 秒")
    
    return h_donors, o_positions

def analyze_frame_hbonds_parallel(frame_data, h_atoms_ids, o_atoms_ids, o_positions, 
                                 h_donors, distance_cutoff=3.5, angle_cutoff=120.0):
    """
    并行版本：分析单帧中的氢键
    """
    frame_idx, frame_time, atom_positions = frame_data
    
    frame_hbonds = []
    angle_cutoff_rad = np.radians(angle_cutoff)
    
    # 为当前帧创建氢原子位置
    h_positions = np.array([atom_positions[h_id-1] for h_id in h_atoms_ids])
    
    # 对每个氢原子
    for i, h_id in enumerate(h_atoms_ids):
        h_pos = h_positions[i]
        
        # 查找供体氧原子
        donor_o_id = None
        min_dist = float('inf')
        
        # 只检查预计算中可能的供体
        for donor_id in h_donors.get(h_id, []):
            try:
                donor_pos = atom_positions[donor_id-1]
                dist = np.linalg.norm(h_pos - donor_pos)
                
                if dist < min_dist:
                    donor_o_id = donor_id
                    min_dist = dist
            except IndexError:
                continue
        
        if donor_o_id is None:
            continue
        
        donor_pos = atom_positions[donor_o_id-1]
        
        # 查找受体氧原子
        # 计算氢原子到所有氧原子的距离
        h_to_o_vectors = o_positions - h_pos
        if CUDA_AVAILABLE:
            # 使用GPU加速计算
            h_to_o_vectors_gpu = cp.asarray(h_to_o_vectors)
            h_to_acc_distances = cp.linalg.norm(h_to_o_vectors_gpu, axis=1)
            h_to_acc_distances = cp.asnumpy(h_to_acc_distances)
        else:
            # 使用CPU计算
            h_to_acc_distances = np.linalg.norm(h_to_o_vectors, axis=1)
        
        # 筛选距离在氢键范围内的氧原子
        for j, (acc_id, dist) in enumerate(zip(o_atoms_ids, h_to_acc_distances)):
            if not (1.5 <= dist <= distance_cutoff) or acc_id == donor_o_id:
                continue
            
            acc_pos = o_positions[j]
            
            # 计算角度
            dh_vector = h_pos - donor_pos
            ha_vector = acc_pos - h_pos
            
            dh_norm = np.linalg.norm(dh_vector)
            ha_norm = np.linalg.norm(ha_vector)
            
            if dh_norm == 0 or ha_norm == 0:
                continue
                
            cos_angle = np.dot(dh_vector, ha_vector) / (dh_norm * ha_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # 氢键角度检查
            if angle >= angle_cutoff_rad:
                # 使用原子ID元组作为氢键标识
                hbond_id = (donor_o_id, h_id, acc_id)
                frame_hbonds.append(hbond_id)
    
    return frame_hbonds

def analyze_trajectory_hbonds_parallel(u, h_atoms, o_atoms, distance_cutoff=3.5, angle_cutoff=120.0):
    """
    使用并行计算分析整个轨迹中的氢键
    """
    print("\n" + "=" * 60)
    print("开始并行分析轨迹中的氢键")
    print("=" * 60)
    print(f"距离阈值: {distance_cutoff} Å")
    print(f"角度阈值: {angle_cutoff}°")
    
    # 获取CPU核心数，用于并行处理
    num_processes = min(mp.cpu_count(), 8)  # 限制最大进程数
    print(f"使用 {num_processes} 个进程进行并行计算")
    
    # 预计算氢键相关信息
    h_donors, o_positions = precompute_donor_info(h_atoms, o_atoms)
    
    # 准备并行处理的数据
    frames_data = []
    frame_times = []
    for frame_idx, ts in enumerate(u.trajectory):
        frame_times.append(ts.time)
        frames_data.append((frame_idx, ts.time, u.atoms.positions.copy()))
    
    # 存储每帧的氢键信息
    all_hbonds = []  # 存储每帧的氢键
    hbond_counts = []  # 存储每帧的氢键数量
    
    # 进度显示相关变量
    total_frames = len(frames_data)
    progress_interval = max(1, total_frames // 10)  # 每10%显示一次进度
    start_time = time.time()
    
    # 使用进程池并行处理帧
    analyze_frame_partial = partial(analyze_frame_hbonds_parallel, 
                                   h_atoms_ids=[h.id for h in h_atoms],
                                   o_atoms_ids=[o.id for o in o_atoms],
                                   o_positions=o_positions,
                                   h_donors=h_donors,
                                   distance_cutoff=distance_cutoff,
                                   angle_cutoff=angle_cutoff)
    
    # 分批处理以显示进度
    batch_size = max(1, total_frames // 10)
    processed_frames = 0
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 分批提交任务
        for i in range(0, total_frames, batch_size):
            batch = frames_data[i:i+batch_size]
            
            # 提交批处理任务
            batch_results = list(executor.map(analyze_frame_partial, batch))
            
            # 收集结果
            for frame_hbonds in batch_results:
                all_hbonds.append(frame_hbonds)
                hbond_counts.append(len(frame_hbonds))
            
            # 更新进度
            processed_frames += len(batch)
            elapsed_time = time.time() - start_time
            progress = processed_frames / total_frames * 100
            eta = (elapsed_time / processed_frames) * (total_frames - processed_frames)
            
            print(f"  进度: {progress:5.1f}% ({processed_frames:4d}/{total_frames}) "
                  f"| 当前批平均氢键数: {np.mean([len(hb) for hb in batch_results]):4.1f} "
                  f"| 已用时间: {elapsed_time:6.1f}s "
                  f"| 预计剩余: {eta:6.1f}s")
    
    total_time = time.time() - start_time
    print(f"轨迹分析完成，总用时: {total_time:.1f} 秒")
    
    return all_hbonds, frame_times, hbond_counts

def analyze_hbond_lifetimes(all_hbonds):
    """
    分析氢键寿命
    """
    if not all_hbonds:
        return None
    
    print("\n分析氢键寿命...")
    
    # 跟踪每个氢键的持续时间
    hbond_durations = defaultdict(list)
    hbond_current_start = {}
    
    for frame_idx, frame_hbonds in enumerate(all_hbonds):
        # 只使用原子索引作为氢键标识 (donor, hydrogen, acceptor)
        current_hbonds = set((hbond[0], hbond[1], hbond[2]) for hbond in frame_hbonds)
        
        # 检查哪些氢键结束了
        ended_hbonds = set(hbond_current_start.keys()) - current_hbonds
        for hbond in ended_hbonds:
            start_frame = hbond_current_start.pop(hbond)
            duration = frame_idx - start_frame
            hbond_durations[hbond].append(duration)
        
        # 检查哪些氢键开始
        new_hbonds = current_hbonds - set(hbond_current_start.keys())
        for hbond in new_hbonds:
            hbond_current_start[hbond] = frame_idx
    
    # 处理持续到轨迹结束的氢键
    for hbond, start_frame in hbond_current_start.items():
        duration = len(all_hbonds) - start_frame
        hbond_durations[hbond].append(duration)
    
    # 计算平均寿命
    all_durations = []
    for durations in hbond_durations.values():
        all_durations.extend(durations)
    
    if not all_durations:
        return None
    
    return {
        'lifetimes': all_durations,
        'avg_lifetime': np.mean(all_durations),
        'max_lifetime': max(all_durations),
        'min_lifetime': min(all_durations),
        'detailed_lifetimes': hbond_durations
    }

def calculate_time_correlation_function_vectorized(all_hbonds, max_tau=None):
    """
    向量化计算氢键的时间关联函数
    """
    if not all_hbonds:
        return None
    
    if max_tau is None:
        max_tau = min(50, len(all_hbonds) // 5)
    
    if len(all_hbonds) <= max_tau:
        return None
    
    print("\n向量化计算氢键时间关联函数...")
    print(f"  最大时间延迟: {max_tau} 帧")
    
    # 获取所有出现过的氢键
    all_unique_hbonds = set()
    for frame_hbonds in all_hbonds:
        all_unique_hbonds.update(frame_hbonds)
    
    if not all_unique_hbonds:
        return None
    
    print(f"  氢键种类数: {len(all_unique_hbonds)}")
    
    # 转换为更高效的数据结构
    n_frames = len(all_hbonds)
    n_hbonds = len(all_unique_hbonds)
    
    # 创建氢键存在矩阵 (氢键种类数 x 帧数)
    hbond_matrix = np.zeros((n_hbonds, n_frames), dtype=bool)
    hbond_list = list(all_unique_hbonds)
    
    # 填充矩阵
    for frame_idx, frame_hbonds in enumerate(all_hbonds):
        frame_set = set(frame_hbonds)
        for hbond_idx, hbond in enumerate(hbond_list):
            if hbond in frame_set:
                hbond_matrix[hbond_idx, frame_idx] = True
    
    # 计算时间关联函数
    taus = np.arange(min(max_tau + 1, n_frames))
    tcf_values = np.zeros(len(taus))
    
    # 向量化计算
    for i, tau in enumerate(taus):
        if tau == 0:
            # t=0时，关联函数为1
            tcf_values[i] = 1.0
        else:
            # 计算所有氢键在时间t和t+tau的相关性
            matrix_t = hbond_matrix[:, :-tau] if tau < n_frames else np.array([])
            matrix_t_tau = hbond_matrix[:, tau:] if tau < n_frames else np.array([])
            
            if matrix_t.size > 0 and matrix_t_tau.size > 0:
                # 计算点积得到同时存在的氢键数
                correlation = np.sum(matrix_t & matrix_t_tau)
                # 计算t时刻存在的氢键总数
                total_t = np.sum(matrix_t)
                
                if total_t > 0:
                    tcf_values[i] = correlation / total_t
                else:
                    tcf_values[i] = 0
            else:
                tcf_values[i] = 0
    
    return {
        'taus': taus.tolist(),
        'tcf_values': tcf_values.tolist()
    }

def plot_hbond_count_vs_time(frame_times, hbond_counts, filename='hbond_count_vs_time.png'):
    """
    绘制氢键数随时间变化图
    """
    plt.figure(figsize=(10, 6))
    plt.plot(frame_times, hbond_counts, 'b-', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('Number of Hydrogen Bonds')
    plt.title('Hydrogen Bond Count vs Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"氢键数随时间变化图已保存为 '{filename}'")

def plot_lifetime_distribution(lifetimes, filename='hbond_lifetime_distribution.png'):
    """
    绘制氢键寿命分布图
    """
    plt.figure(figsize=(10, 6))
    plt.hist(lifetimes, bins=50, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
    plt.xlabel('Hydrogen Bond Lifetime (frames)')
    plt.ylabel('Frequency')
    plt.title('Hydrogen Bond Lifetime Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"氢键寿命分布图已保存为 '{filename}'")

def plot_time_correlation_function(taus, tcf_values, filename='hbond_time_correlation_function.png'):
    """
    绘制时间关联函数图
    """
    plt.figure(figsize=(10, 6))
    plt.plot(taus, tcf_values, 'r-', linewidth=2)
    plt.xlabel('Time Lag (frames)')
    plt.ylabel('Hydrogen Bond Correlation Function')
    plt.title('Hydrogen Bond Time Correlation Function')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"氢键时间关联函数图已保存为 '{filename}'")

def save_results(frame_times, hbond_counts, lifetime_results, tcf_results, prefix='hbond'):
    """
    保存分析结果到文件
    """
    # 保存氢键数随时间变化数据
    np.savetxt(f'{prefix}_counts.dat', 
               np.column_stack([frame_times, hbond_counts]), 
               header='Time HbondCount', 
               fmt='%.2f %d')
    print(f"氢键数数据已保存为 '{prefix}_counts.dat'")
    
    # 保存寿命分析结果
    if lifetime_results:
        with open(f'{prefix}_lifetimes.dat', 'w') as f:
            f.write('Lifetime(frames)\n')
            for lifetime in lifetime_results['lifetimes']:
                f.write(f'{lifetime}\n')
        print(f"氢键寿命数据已保存为 '{prefix}_lifetimes.dat'")
    
    # 保存时间关联函数结果
    if tcf_results:
        np.savetxt(f'{prefix}_tcf.dat',
                   np.column_stack([tcf_results['taus'], tcf_results['tcf_values']]),
                   header='TimeLag(frames) CorrelationFunction',
                   fmt='%d %.6f')
        print(f"氢键时间关联函数数据已保存为 '{prefix}_tcf.dat'")

def display_statistics(all_hbonds, frame_times, hbond_counts):
    """
    显示氢键统计信息
    """
    print("\n" + "=" * 60)
    print("氢键统计分析结果")
    print("=" * 60)
    
    # 基本统计
    total_hbonds = sum(hbond_counts)
    print(f"轨迹分析结果:")
    print(f"  总帧数        : {len(all_hbonds):8d}")
    print(f"  总氢键数      : {total_hbonds:8d}")
    print(f"  平均每帧氢键数: {np.mean(hbond_counts):8.1f}")
    print(f"  氢键数标准差  : {np.std(hbond_counts):8.1f}")
    
    if hbond_counts:
        print(f"  最大氢键数    : {max(hbond_counts):8d} (第{hbond_counts.index(max(hbond_counts))+1}帧)")
        print(f"  最小氢键数    : {min(hbond_counts):8d} (第{hbond_counts.index(min(hbond_counts))+1}帧)")

def analyze_hbond_pairs(all_hbonds, u):
    """
    分析不同供体和受体对之间的氢键统计
    """
    if not all_hbonds:
        return None
    
    # 统计每种氢键对的数量
    hbond_pair_counts = Counter()
    
    # 收集所有氢键对信息
    for frame_hbonds in all_hbonds:
        for donor_id, h_id, acceptor_id in frame_hbonds:
            # 创建供体-受体对标识
            pair_id = (donor_id, acceptor_id)
            hbond_pair_counts[pair_id] += 1
    
    return hbond_pair_counts

def display_hbond_pairs(hbond_pair_counts, u, type_labels=None, max_pairs=20):
    """
    显示氢键对统计结果
    """
    if not hbond_pair_counts:
        print("没有氢键对统计结果")
        return
    
    print("\n" + "=" * 60)
    print("氢键供体-受体对统计分析结果")
    print("=" * 60)
    
    # 按出现频率排序
    sorted_pairs = hbond_pair_counts.most_common()
    
    print(f"共发现 {len(sorted_pairs)} 种不同的供体-受体对")
    print(f"显示前 {min(max_pairs, len(sorted_pairs))} 种最常见氢键对:")
    print(f"{'排名':<4} {'供体类型':<8} {'供体标签':<10} {'供体名称':<10} {'受体类型':<8} {'受体标签':<10} {'受体名称':<10} {'出现次数':<8} {'占总数比(%)':<10}")
    print("-" * 100)
    
    total_count = sum(hbond_pair_counts.values())
    
    for i, ((donor_id, acceptor_id), count) in enumerate(sorted_pairs[:max_pairs]):
        # 获取原子信息
        donor = None
        acceptor = None
        
        if u and hasattr(u, 'atoms') and u.atoms is not None:
            if donor_id > 0 and donor_id <= len(u.atoms):
                donor = u.atoms[donor_id-1]
            if acceptor_id > 0 and acceptor_id <= len(u.atoms):
                acceptor = u.atoms[acceptor_id-1]
        
        if donor is not None and acceptor is not None:
            # 使用get_atom_name函数获取原子名称
            donor_name = get_atom_name(donor, type_labels)
            acceptor_name = get_atom_name(acceptor, type_labels)
            
            # 获取供体和受体的标签
            donor_label = type_labels.get(donor.type, 'N/A') if type_labels else 'N/A'
            acceptor_label = type_labels.get(acceptor.type, 'N/A') if type_labels else 'N/A'
            
            percentage = (count / total_count) * 100
            print(f"{i+1:<4} {donor.type:<8} {donor_label:<10} {donor_name:<10} {acceptor.type:<8} {acceptor_label:<10} {acceptor_name:<10} {count:<8} {percentage:<10.2f}")
        else:
            percentage = (count / total_count) * 100
            print(f"{i+1:<4} {donor_id:<8} {'N/A':<10} {'N/A':<10} {acceptor_id:<8} {'N/A':<10} {'N/A':<10} {count:<8} {percentage:<10.2f}")

def read_atom_type_labels(data_file):
    """
    从LAMMPS data文件中读取Atom Type Labels信息
    """
    type_labels = {}
    
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        # 查找Atom Type Labels部分
        in_labels_section = False
        in_masses_section = False
        for line in lines:
            line = line.strip()
            
            # 处理Atom Type Labels部分
            if line == "Atom Type Labels":
                in_labels_section = True
                in_masses_section = False
                continue
            
            # 处理Masses部分
            if line == "Masses":
                in_labels_section = False
                in_masses_section = True
                continue
            
            # 结束条件：空行或遇到其他节标题
            if not line:
                in_labels_section = False
                in_masses_section = False
                continue
                
            if line.endswith("Labels") and line != "Atom Type Labels":
                in_labels_section = False
                continue
            
            if line.endswith("Coeffs"):
                in_masses_section = False
                continue
            
            # 解析Atom Type Labels部分
            if in_labels_section:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        atom_type = int(parts[0])
                        label = parts[1]
                        type_labels[atom_type] = label
                    except ValueError:
                        # 跳过无法解析的行
                        continue
            
            # 解析Masses部分的注释
            if in_masses_section and '#' in line:
                # 匹配类似 "1 1.0080  # H_" 这样的行
                parts = line.split('#')
                if len(parts) >= 2:
                    mass_part = parts[0].strip()
                    label_part = parts[1].strip()
                    mass_parts = mass_part.split()
                    if len(mass_parts) >= 1:
                        try:
                            atom_type = int(mass_parts[0])
                            # 只有当Atom Type Labels部分没有定义时才使用Masses部分的注释
                            if atom_type not in type_labels:
                                type_labels[atom_type] = label_part
                        except ValueError:
                            # 跳过无法解析的行
                            continue
    except Exception as e:
        print(f"警告: 无法读取Atom Type Labels信息: {e}")
    
    return type_labels

def get_atom_name(atom, type_labels=None):
    """
    获取原子名称，优先级：
    1. Atom Type Labels映射的标签
    2. 原子自身的名称属性
    3. 类型信息
    """
    if type_labels and hasattr(atom, 'type'):
        # 使用Atom Type Labels映射
        label = type_labels.get(atom.type)
        if label:
            return label
    
    try:
        # 尝试使用原子自身的名称
        return atom.name
    except:
        # 使用类型信息
        return f"Type_{atom.type}"

def get_atom_selection_info(u):
    """
    获取系统中原子类型和名称信息
    """
    print("\n" + "-" * 50)
    print("系统原子类型和名称信息")
    print("-" * 50)
    
    # 统计原子类型
    type_info = {}
    name_info = {}
    
    for atom in u.atoms:
        # 类型统计
        if atom.type not in type_info:
            type_info[atom.type] = {'count': 0, 'elements': set()}
        type_info[atom.type]['count'] += 1
        if hasattr(atom, 'element') and atom.element is not None:
            type_info[atom.type]['elements'].add(atom.element)
        
        # 名称统计
        if atom.name not in name_info:
            name_info[atom.name] = {'count': 0, 'types': set()}
        name_info[atom.name]['count'] += 1
        name_info[atom.name]['types'].add(atom.type)
    
    print("原子类型信息:")
    for atom_type in sorted(type_info.keys()):
        info = type_info[atom_type]
        elements = ', '.join(sorted(info['elements'])) if info['elements'] else 'N/A'
        print(f"  类型 {atom_type}: {info['count']:6d} 个原子 (元素: {elements})")
    
    print("\n原子名称信息 (部分):")
    name_items = list(name_info.items())[:20]  # 只显示前20个
    for name, info in name_items:
        types = ', '.join(sorted(info['types']))
        print(f"  名称 {name}: {info['count']:6d} 个原子 (类型: {types})")
    
    if len(name_info) > 20:
        print(f"  ... 还有 {len(name_info) - 20} 种名称未显示")
    
    return type_info, name_info

def main():
    """
    主函数
    """
    try:
        # 获取用户输入
        topology_file, trajectory_file, timestep, distance_cutoff, angle_cutoff = get_user_input()
        
        # 加载系统和轨迹
        print("\n加载系统和轨迹文件...")
        u = mda.Universe(topology_file, trajectory_file, format='LAMMPSDUMP',dt=timestep)
        print("文件加载成功!")
        
        # 读取Atom Type Labels信息
        type_labels = read_atom_type_labels(topology_file)
        if type_labels:
            print(f"从data文件读取到 {len(type_labels)} 个Atom Type Labels")
        else:
            print("未从data文件读取到Atom Type Labels信息")
        
        # 选择原子类型
        h_atoms, o_atoms = select_atom_types(u, type_labels)
        
        # 询问氢键分析类型
        print("\n" + "-" * 50)
        print("氢键分析类型选择:")
        print("1. 总体统计 (不区分供体-受体对)")
        print("2. 详细统计 (区分供体-受体对)")
        print("3. 两者都选")
        analysis_choice = input("请选择分析类型 (默认 1): ").strip() or "1"
        
        # 确认开始分析
        print("\n" + "=" * 60)
        user_confirm = input("确认开始分析轨迹? (y/N): ").strip().lower()
        if user_confirm not in ['y', 'yes']:
            print("用户取消分析。")
            return
        
        # 分析轨迹中的氢键（使用并行优化版本）
        all_hbonds, frame_times, hbond_counts = analyze_trajectory_hbonds_parallel(
            u, h_atoms, o_atoms, distance_cutoff, angle_cutoff)
        
        # 显示统计信息
        display_statistics(all_hbonds, frame_times, hbond_counts)
        
        # 根据选择进行详细分析
        hbond_pair_results = None
        if analysis_choice in ["2", "3"]:
            print("\n分析供体-受体对...")
            hbond_pair_results = analyze_hbond_pairs(all_hbonds, u)
            display_hbond_pairs(hbond_pair_results, u, type_labels)
            
            # 询问是否显示原子选择信息
            show_atom_info = input("\n是否显示详细的原子类型和名称信息? (y/N): ").strip().lower()
            if show_atom_info in ['y', 'yes']:
                get_atom_selection_info(u)
        
        # 绘制氢键数随时间变化图
        print("\n绘制氢键数随时间变化图...")
        plot_hbond_count_vs_time(frame_times, hbond_counts)
        
        # 分析氢键寿命
        lifetime_results = analyze_hbond_lifetimes(all_hbonds)
        if lifetime_results:
            print(f"氢键寿命分析结果:")
            print(f"  平均氢键寿命: {lifetime_results['avg_lifetime']:6.2f} 帧")
            print(f"  最长氢键寿命: {lifetime_results['max_lifetime']:6d} 帧")
            print(f"  最短氢键寿命: {lifetime_results['min_lifetime']:6d} 帧")
            plot_lifetime_distribution(lifetime_results['lifetimes'])
        
        # 询问是否计算时间关联函数
        print("\n" + "-" * 50)
        calc_tcf = input("是否计算氢键时间关联函数? (y/N): ").strip().lower()
        tcf_results = None
        if calc_tcf in ['y', 'yes']:
            max_tau = input("最大时间延迟 (默认为帧数的1/5): ").strip()
            max_tau = int(max_tau) if max_tau else None
            tcf_results = calculate_time_correlation_function_vectorized(all_hbonds, max_tau)
            if tcf_results:
                plot_time_correlation_function(tcf_results['taus'], tcf_results['tcf_values'])
        
        # 保存结果
        print("\n保存分析结果...")
        save_results(frame_times, hbond_counts, lifetime_results, tcf_results, 'template_based_optimized_hbond')
        
        # 保存氢键对结果
        if hbond_pair_results:
            with open('hbond_pairs.dat', 'w') as f:
                f.write('DonorID DonorType DonorLabel DonorName AcceptorID AcceptorType AcceptorLabel AcceptorName Count\n')
                for (donor_id, acceptor_id), count in hbond_pair_results.most_common():
                    # 获取原子信息
                    donor = None
                    acceptor = None
                    
                    if u and hasattr(u, 'atoms') and u.atoms is not None:
                        if donor_id > 0 and donor_id <= len(u.atoms):
                            donor = u.atoms[donor_id-1]
                        if acceptor_id > 0 and acceptor_id <= len(u.atoms):
                            acceptor = u.atoms[acceptor_id-1]
                    
                    if donor is not None and acceptor is not None:
                        donor_name = get_atom_name(donor, type_labels)
                        acceptor_name = get_atom_name(acceptor, type_labels)
                        donor_label = type_labels.get(donor.type, 'N/A') if type_labels else 'N/A'
                        acceptor_label = type_labels.get(acceptor.type, 'N/A') if type_labels else 'N/A'
                        f.write(f'{donor_id} {donor.type} {donor_label} {donor_name} {acceptor_id} {acceptor.type} {acceptor_label} {acceptor_name} {count}\n')
                    else:
                        f.write(f'{donor_id} N/A N/A N/A {acceptor_id} N/A N/A N/A {count}\n')
            print(f"氢键对统计数据已保存为 'hbond_pairs.dat'")
        
        print("\n" + "=" * 60)
        print("氢键分析完成!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except KeyboardInterrupt:
        print("\n\n用户中断分析。")
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()