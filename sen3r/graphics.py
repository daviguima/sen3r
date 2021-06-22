# TODO: this is very specific, make it more generic.
def plot_s3_lv2_reflectances(self, radiance_list, icor, band_radiances, figure_title):
    # TODO: write docstrings
    ### L2 WFR
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Wavelenght (nm)')
    ax1.set_ylabel('Reflectance')
    ax1.set_title(figure_title, y=1, fontsize=16)
    ax1.plot(list(dd.s3_bands_l2.values()), icor, label='L1 iCOR', marker='o')
    ax1.plot(list(dd.s3_bands_l2.values()), band_radiances, label='L2 WFR', marker='o')
    ax1.axhline(y=0, xmin=0, xmax=1, linewidth=0.5, color='black', linestyle='--')
    ax1.set_xticks(list(dd.s3_bands_l2.values()))
    ax1.set_xticklabels(list(dd.s3_bands_l2.values()))
    ax1.tick_params(labelrotation=90, labelsize='small')
    # ax1.set_yticklabels(labels=np.linspace(
    #     ax1.get_yticks().min(), ax1.get_yticks().max(), len(ax1.get_yticks()) * 2),
    #     rotation=0)
    ax1.legend()
    ax2 = ax1.twiny()
    ax2.plot(np.linspace(min(list(dd.s3_bands_l2.values())),
                         max(list(dd.s3_bands_l2.values())),
                         num=len(dd.s3_bands_l2)), band_radiances, alpha=0.0)
    ax2.set_xticks(list(dd.s3_bands_l2.values()))
    ax2.set_xticklabels(list(dd.s3_bands_l2.keys()))
    ax2.tick_params(labelrotation=90, labelsize='xx-small')
    ax2.set_title('Sentinel-3 Oa Bands', y=0.93, x=0.12, fontsize='xx-small')
    # ax2.grid()
    plt.show()


@staticmethod  # TODO: as the name states, temp stuff either needs to be fixed, moved or removed.
def _temp_plot(lon, lat, plot_var, roi_lon=None, roi_lat=None):
    # Miller projection:
    m = Basemap(projection='mill',
                lat_ts=10,
                llcrnrlon=lon.min(),
                urcrnrlon=lon.max(),
                llcrnrlat=lat.min(),
                urcrnrlat=lat.max(),
                resolution='c')

    # BR bbox
    # m = Basemap(projection='mill',
    #             llcrnrlat=-60,
    #             llcrnrlon=-90,
    #             urcrnrlat=20,
    #             urcrnrlon=-25)

    x, y = m(lon, lat)
    # x, y = m(lon, lat, inverse=True)

    m.pcolormesh(x, y, plot_var, shading='flat', cmap=plt.cm.jet)
    m.colorbar(location='right')  # ('top','bottom','left','right')

    # dd_lon, dd_lat = -60.014493, -3.158980  # Manaus
    # if roi_lon is not None and roi_lat is not None:
    xpt, ypt = m(roi_lon, roi_lat)
    m.plot(xpt, ypt, 'rD')  # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    # m.drawcoastlines()
    plt.show()
    # plt.figure()