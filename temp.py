fig, ax1 = plt.subplots()
ax1.set_xlabel('Wavelenght (nm)')
ax1.set_ylabel('Reflectance')
ax1.plot(list(s3_bands_l2.values()), icor, label='L1 iCOR', marker='o')
ax1.plot(list(s3_bands_l2.values()), band_radiances, label='L2 WFR', marker='o')
ax1.axhline(y=0, xmin=0, xmax=1, linewidth=0.5, color='black', linestyle='--')
ax1.set_title('AX1 - Title', y=1, fontsize=16)
ax1.set_xticks(list(s3_bands_l2.values()))
ax1.set_xticklabels(list(s3_bands_l2.values()))
ax1.tick_params(labelrotation=90, labelsize='small')

# ax1.set_yticklabels(labels=np.linspace(
#     ax1.get_yticks().min(), ax1.get_yticks().max(), len(ax1.get_yticks()) * 2),
#     rotation=0)

ax1.legend()

ax2 = ax1.twiny()
ax2.plot(np.linspace(min(list(s3_bands_l2.values())),
                     max(list(s3_bands_l2.values())),
                     num=len(s3_bands_l2)), band_radiances, alpha=0.0)
ax2.set_xticks(list(s3_bands_l2.values()))
ax2.set_xticklabels(list(s3_bands_l2.keys()))
ax2.tick_params(labelrotation=90, labelsize='xx-small')
# ax2.grid()

ax2.set_title('Sentinel-3 Oa Bands', y=0.93, x=0.12, fontsize='xx-small')
# fig.suptitle('This is a somewhat long figure title', fontsize=16)
# fig.subplots_adjust(top=0.5)
# fig.set_tight_layout(True)
# plt.tight_layout()
plt.show()