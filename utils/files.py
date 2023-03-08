import yaml

def join(loader, node):
    """python加载yaml时，考虑锚点值
    usage：
    #    yaml.add_constructor('!join', join)
    #    yaml.load("""
    #    user_dir: &DIR /home/user
    #    user_pics: !join [*DIR, /pics]
    """)   ——>  {'user_dir': '/home/user', 'user_pics': '/home/user/pics'}
    """
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def update_arguments(args, config, update: bool = False):
    """update config to args
    """
    for key, value in config.items():
        # 对于args中设置的值为最终值,即使config里面有冲突的值,仍以args中的参数值为准
        if key in args:
            print(f"该参数{key}的原值为{value}, 新值为{args.__dict__[key]}")
            if not update:
                continue
        args.__setattr__(key, value)
    return args