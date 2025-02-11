import torch
import matplotlib.pyplot as plt

def value_map(x, from_min, from_max, to_min, to_max):
    """
    x를 [from_min, from_max] 구간에서 [to_min, to_max] 구간으로 선형 매핑합니다.
    """
    return (x - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

def simulate_sampling(content_or_style, schedule_type='linear', batch_size=10000, num_train_timesteps=1000, 
                      min_noise_steps=0, max_noise_steps=1000, device=torch.device("cpu"),
                      content_bias=1.0):
    """
    주어진 content_or_style ('content', 'style', 'balanced')에 따라 timestep 인덱스를 계산하고,
    schedule_type에 따라 타임스텝 스케줄을 생성합니다.
    
    - content의 경우, content_bias (0이면 균일, 1이면 full cubic sampling) 값을 사용하여
      원래의 균등 난수와 cubic 변환값을 보간합니다.
    - style의 경우, (1 - x^3)로 높은 timestep을 선호합니다.
    - balanced의 경우, 단순 정수 샘플링을 수행합니다.
    
    그리고 schedule_type에 따라 타임스텝 스케줄을 생성합니다.
      - 'linear': torch.linspace(1000, 0, num_train_timesteps, device=device)
      - 'sigmoid': torch.sigmoid를 이용한 방식 (내림차순 정렬)
    
    Returns:
        orig_timesteps (torch.Tensor): [0,1] 균등 난수들
        timestep_indices (torch.Tensor): 변환된 timestep 인덱스들 (정수형)
        timesteps (torch.Tensor): schedule에서 선택된 최종 타임스텝 값들
    """
    # 원래의 난수 생성 (이 값은 나중에 매핑 관계를 살펴보기 위함)
    orig_timesteps = torch.rand((batch_size,), device=device)
    
    if content_or_style in ['content', 'style']:
        if content_or_style == 'content':
            # content의 경우, content_bias로 균일분포와 cubic sampling(x^3) 사이를 보간
            adjusted_timesteps = (1 - content_bias) * orig_timesteps + content_bias * (orig_timesteps ** 3)
            timestep_indices = adjusted_timesteps * num_train_timesteps
        elif content_or_style == 'style':
            timestep_indices = (1 - orig_timesteps ** 3) * num_train_timesteps

        # value_map을 이용해 [0, num_train_timesteps-1] 구간을 [min_noise_steps, max_noise_steps-1]로 매핑
        timestep_indices = value_map(
            timestep_indices, 
            0, 
            num_train_timesteps - 1, 
            min_noise_steps, 
            max_noise_steps - 1
        )
        # 정수형 변환 후 클램핑
        timestep_indices = timestep_indices.long().clamp(min_noise_steps + 1, max_noise_steps - 1)
        
    elif content_or_style == 'balanced':
        if min_noise_steps == max_noise_steps:
            timestep_indices = torch.ones((batch_size,), device=device) * min_noise_steps
        else:
            timestep_indices = torch.randint(
                min_noise_steps + 1, 
                max_noise_steps - 1, 
                (batch_size,), 
                device=device
            )
        timestep_indices = timestep_indices.long()
    else:
        raise ValueError(f"Unknown content_or_style {content_or_style}")
    
    # 타임스텝 스케줄 생성 부분: schedule_type에 따라 다르게 생성
    if schedule_type == 'linear':
        schedule = torch.linspace(1000, 0, num_train_timesteps, device=device)
    elif schedule_type == 'sigmoid':
        # 예시: 난수를 생성하여 sigmoid를 적용한 후 1000~0 범위로 스케일하고 내림차순 정렬
        t = torch.sigmoid(torch.randn((num_train_timesteps,), device=device))
        schedule = (1 - t) * 1000.0
        schedule, _ = torch.sort(schedule, descending=True)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")
    
    # 생성한 schedule에서, timestep_indices를 인덱스로 사용하여 최종 타임스텝 선택
    timesteps = torch.stack([schedule[idx.item()] for idx in timestep_indices])
    
    return orig_timesteps, timestep_indices, timesteps

def visualize_sampling(orig_timesteps, timestep_indices, timesteps, title_prefix=""):
    """
    히스토그램과 산점도를 통해 결과를 시각화합니다.
    """
    # 히스토그램: timestep 인덱스 분포
    plt.figure(figsize=(12, 5))
    plt.hist(timestep_indices.cpu().numpy(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Timestep Index')
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix} Distribution of Timestep Indices')
    plt.grid(True)
    plt.show()

    # 산점도: 원래 난수와 매핑된 인덱스
    plt.figure(figsize=(12, 5))
    plt.scatter(orig_timesteps.cpu().numpy(), timestep_indices.cpu().numpy(), alpha=0.3, color='coral')
    plt.xlabel('Original Timesteps (0 to 1)')
    plt.ylabel('Timestep Index')
    plt.title(f'{title_prefix} Mapping: Original Timesteps -> Timestep Indices')
    plt.grid(True)
    plt.show()

def main():
    batch_size = 10000
    num_train_timesteps = 1000
    min_noise_steps = 0
    max_noise_steps = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_or_style = 'content'
    schedule_type = 'sigmoid'
    content_bias = 0.7

    orig, indices, ts = simulate_sampling(
        content_or_style=content_or_style, 
        schedule_type=schedule_type,
        batch_size=batch_size, 
        num_train_timesteps=num_train_timesteps,
        min_noise_steps=min_noise_steps,
        max_noise_steps=max_noise_steps,
        device=device,
        content_bias=content_bias
    )
    # print(f"--- {option.upper()} with {schedule_type.upper()} schedule ---")
    print(f"timestep_indices: min {indices.min().item()}, max {indices.max().item()}")
    print(f"timesteps (from schedule): min {ts.min().item():.2f}, max {ts.max().item():.2f}\n")
    visualize_sampling(orig, indices, ts, title_prefix=f"{content_or_style} ({schedule_type})")

    # schedule_type: 'linear'와 'sigmoid' 두 가지 경우를 테스트
    # for schedule_type in ['linear', 'sigmoid']:
    #     for option in ['content', 'style', 'balanced']:
    #         # 예시로 content_bias는 0.0 (균일)로 설정합니다.
    #         orig, indices, ts = simulate_sampling(
    #             content_or_style=option, 
    #             schedule_type=schedule_type,
    #             batch_size=batch_size, 
    #             num_train_timesteps=num_train_timesteps,
    #             min_noise_steps=min_noise_steps,
    #             max_noise_steps=max_noise_steps,
    #             device=device,
    #             content_bias=0.0
    #         )
    #         print(f"--- {option.upper()} with {schedule_type.upper()} schedule ---")
    #         print(f"timestep_indices: min {indices.min().item()}, max {indices.max().item()}")
    #         print(f"timesteps (from schedule): min {ts.min().item():.2f}, max {ts.max().item():.2f}\n")
    #         visualize_sampling(orig, indices, ts, title_prefix=f"{option.capitalize()} ({schedule_type})")

if __name__ == '__main__':
    main()
