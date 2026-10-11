def plot(fx, x, title):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(' matplotlib is not installed; skipping the string energy plot (pip install pygsm[plot])')
        return
    plt.figure(1)
    plt.title(f"String {title:04d}")
    plt.plot(x, fx, color='b', label='Energy', linewidth=2, marker='o', markersize=12)
    plt.xlabel('Node Number')
    plt.ylabel('Energy (kcal/mol)')
    plt.legend(loc='best')
    plt.savefig(f'{title:04d}_string.png', dpi=600)


