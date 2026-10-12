def plot(fx, x, title):
    """Save the string energy profile to ``string_<title>.png`` (skipped without matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(' matplotlib is not installed; skipping the string energy plot (pip install pygsm[plot])')
        return
    fig, ax = plt.subplots()
    ax.set_title(f'String {title:03d}')
    ax.plot(x, fx, color='b', label='Energy', linewidth=2, marker='o', markersize=12)
    ax.set_xlabel('Node Number')
    ax.set_ylabel('Energy (kcal/mol)')
    ax.legend(loc='best')
    fig.savefig(f'string_{title:03d}.png', dpi=200)
    plt.close(fig)
