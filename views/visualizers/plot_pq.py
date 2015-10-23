import logging
import sys
import copy
import os
import io
import pickle

from time import gmtime, strftime,asctime
from multiprocessing import Process
import matplotlib as mpl
# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator, NullLocator, MultipleLocator, IndexLocator, FixedLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy import array, linspace, meshgrid, reshape, argmin, arange, append, zeros, ceil, place, ma
from scipy.interpolate import griddata

import HTML
import StringIO
import ho.pisa as pisa
import git

'''
from matplotlib.font_manager import FontProperties
font0 = FontProperties()
alignment = {'horizontalalignment':'center', 'verticalalignment':'baseline'}
family = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
font0.set_family('sans-serif')
'''
### abstract class to define plot viewers
class ImageViewer(object):

    DPI = 150
    SAVE_ALONE = True

    LABEL_FONT_SIZE = 10
    TITLE_FONT_SIZE = 26
    
    DIMS = 2
    
    @staticmethod
    def render(dictionary):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
    @staticmethod
    def save_fig(figure, filename, DPI):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    @staticmethod
    def get_attributes(name):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    @staticmethod
    def get_default_attributes():
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')

    @staticmethod
    def add_inner_title(ax, title, loc, size=None, **kwargs):
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.patheffects import withStroke
        if size is None:
            size = dict(size=22)
            #size = dict(size=mpl.pyplot.rcParams['legend.fontsize'])
        at = AnchoredText(title, loc=loc, prop=size,
                          pad=0., borderpad=0.5,
                          frameon=False, **kwargs)
        ax.add_artist(at)
        at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
        return at
                                  
    @staticmethod
    def render_2d(figure, d, graph_dict, title, fitness, data=None, minVal=None, maxVal=None, float=None, z_label=None):
        if False:
            plot, save_fig = ImageViewer.figure_wrapper(figure, graph_dict, d, title, False)
        else: ## thre d
            plot, save_fig = ImageViewer.figure_wrapper(figure, graph_dict, d, title)
            ### User settings
            font_size = int(graph_dict['font size'])
            plot.set_ylabel('\n' + fitness.get_y_axis_name(), linespacing=3.,
                            fontsize=font_size+12)
            plot.set_xlabel('\n' + fitness.get_x_axis_name(), linespacing=3.,
                            fontsize=font_size+12)

            colour_map = mpl.pyplot.get_cmap("gist_yarg")#graph_dict['colour map'])

            ### Other settings
            locator = LinearLocator(5)
            #locator.tick_values(fitness.designSpace[0]["min"], fitness.designSpace[0]["max"])
            plot.w_xaxis.set_major_locator(locator)
            locator = LinearLocator(6)
            #locator.tick_values(fitness.designSpace[1]["min"], fitness.designSpace[1]["max"])
            plot.w_yaxis.set_major_locator(locator)

            locator = LinearLocator(6)
            plot.w_zaxis.set_major_locator(locator)
            
            
            plot.tick_params(which='major', labelsize=20, axis='both', )
            if fitness.rotate:
                plot.view_init(azim=45)
                if True:
                    plot.w_yaxis.set_major_formatter(FormatStrFormatter('%.2f        '))
                    plot.w_zaxis.set_major_formatter(FormatStrFormatter('%.2f        '))
                    plot.w_xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                else:
                    plot.w_yaxis.set_major_formatter(FormatStrFormatter('%d          '))
                    plot.w_zaxis.set_major_formatter(FormatStrFormatter('%d          '))
                if z_label is None:
                    plot.set_zlabel("\n" + fitness.get_z_axis_name(), linespacing = 3.5, fontsize=font_size+8)
                else:
                    plot.set_zlabel("\n" + z_label, linespacing = 3.5, fontsize=font_size+8)
                plot.get_zticklabels()[-1].set_visible(False)
                plot.get_zticklabels()[-1].set_fontsize(0)
                plot.get_yticklabels()[-1].set_visible(False)
                plot.get_yticklabels()[-1].set_fontsize(0)
            else:
                plot.w_xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                plot.w_yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                plot.w_zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                plot.get_yticklabels()[0].set_visible(False)
                plot.get_yticklabels()[0].set_fontsize(0)
                plot.get_yticklabels()[-1].set_visible(False)
                plot.get_yticklabels()[-1].set_fontsize(0)
                if z_label is None:
                    plot.set_zlabel('\n' + fitness.get_z_axis_name(), linespacing=3.5, fontsize=font_size+8)
                else:
                    plot.set_zlabel("\n" + z_label, linespacing=3.5, fontsize=font_size+8)
                    
            ### Data
            if not (data is None):        
                datamax = data.max()
                logging.debug("Data passed...")
                logging.debug("Prediction done passed...")
                try:
                    
                    zi = griddata((d['x'], d['y']), data,
                                  (d['xi'][None, :], d['yi'][:, None]), method='nearest')

                    #norm = mpl.pyplot.matplotlib.colors.Normalize(data)

                    if minVal and maxVal:
                        #surf = plot.plot_surface(d['X'], d['Y'], zi, rstride=1, cstride=1,
                        surf = plot.contourf(d['X'], d['Y'], zi, rstride=1, cstride=1,
                                             linewidth=0.05, antialiased=True,
                                             cmap=colour_map, alpha = 1.0)#, vmin=minVal, vmax=maxVal)
                    else:
                        surf = plot.plot_surface(d['X'], d['Y'], zi, rstride=1, cstride=1,
                                             linewidth=0.05, antialiased=True,
                                             cmap=colour_map, alpha = 1.0)#, vmin=fitness.minVal, vmax=fitness.maxVal)
                    try:
                        logging.info("Transforming Axis Labels")
                        x_func, y_func = ImageViewer.my_formatter(fitness)
                        plot.w_xaxis.set_major_formatter(mpl.ticker.FuncFormatter(x_func))
                        plot.w_yaxis.set_major_formatter(mpl.ticker.FuncFormatter(y_func))
                    except Exception,e:
                        def two_digits(x, pos): ## its dirty...
                            return '%d'   % ceil(x)
                        plot.w_xaxis.set_major_formatter(mpl.ticker.FuncFormatter(two_digits))
                        plot.w_yaxis.set_major_formatter(mpl.ticker.FuncFormatter(two_digits))
                    if fitness.rotate:
                        def two_digits(x, pos):
                            return '%d' % ceil(x)
                                
                    else:
                        def two_digits(x, pos):
                            return '   %.3g' % ceil(x)
                    if (minVal is None):
                        plot.w_zaxis.set_major_formatter(mpl.ticker.FuncFormatter(two_digits))
                    for zlabel in plot.get_zticklabels():
                        zlabel.set_fontsize(font_size+6)
                    plot.axis('tight') 
                    save_fig()
                except TypeError,e:
                    logging.error('Could not create ' + str(title) + ' plot for the GPR plot: ' + str(e))

    @staticmethod
    def my_formatter(fitness):
        x_func, y_func = fitness.transformation_function_image()
        def mjrXFormatter(x, pos):
            return str(x_func(x))
        def mjrYFormatter(x, pos):
            return str(y_func(x))
        return mjrXFormatter, mjrYFormatter
                
    @staticmethod
    def render_3_4d(figure, info, d, graph_dict, title, data=None, fitness = None, meta_plots=None, lognorm = False, levels=None, mask=False, minVal=None, maxVal=None, cmap_string = "jet", alpha=1.0):
        #lognorm= False
        #plot, save_fig  = ImageViewer.figure_wrapper(figure, graph_dict, d, title, three_d=False)
        ### User settings
        #font_size = int(graph_dict['font size'])
        D_0 = len(data)
        D_1 = len(data[0])
        colour_map = mpl.cm.get_cmap(cmap_string)#gist_yarg")#graph_dict['colour map'])
        colour_map = mpl.cm.get_cmap("YlOrRd")
        data = array(data)
        lognorm = False
        #lognorm = False
        if lognorm:
            place(data,data <=0.,[0.0000001]) 
        if fitness:
            data_max = fitness.maxVal
            ### Data
            if lognorm:
                data_min = max(fitness.minVal,0.0000001)
            else:
                data_min = fitness.minVal
        elif minVal: ## could be a bug if no maxVal... fuck it
            data_max=maxVal
            if lognorm:
                data_min = max(minVal,0.0000001)
            else:
                data_min = minVal
        else:
            data_max=data.max()
            if lognorm:
                data_min = max(data.min(),0.0000001)
            else:
                data_min=data.min()
                
        if not (data is None):        
            logging.debug("Data passed...")
            try:
                fig = pyplot.figure(1,(24., 24.))
                grid = ImageGrid(fig, 111, # similar to subplot(111)
                                nrows_ncols = (D_1, D_0), # creates 2x2 grid of axes
                                axes_pad=0.20, # pad between axes in inch.
                                #label_mode = "L",
                                share_all = True,
                                #aspect=1.0,
                                cbar_mode = 'single',
                                cbar_location = 'bottom',
                                cbar_size=".6%",
                                )
                if lognorm:
                    log_norm = mpl.colors.LogNorm(vmax=data_max,vmin=data_min)
                    # Currently not used - linear scaling
                    linear_norm = mpl.colors.Normalize()
                
                for i in range(D_0):
                    for j in range(D_1):
                        method = 'nearest'
                        
                        zi = griddata((d['x'], d['y']), data[i][j],
                                  (d['xi'][None, :], d['yi'][:, None]), method=method)
                                  
                        if mask:
                            maski = griddata((d['x'], d['y']), mask[i][j],
                                  (d['xi'][None, :], d['yi'][:, None]), method=method)
                            zi = ma.masked_where(maski != 0, zi)
                        #CS = grid[i*D_1 + j].contour(d['X'], d['Y'], zi,colors='k',vmin=data_min,vmax=data_max)                   
                        
                        if meta_plots: ## OLD METAPLOTS
                            colour_map = mpl.cm.get_cmap("gist_yarg")
                            CS = grid[i*D_1 + j].contourf(d['X'], d['Y'], zi,cmap=colour_map,vmin=data_min,vmax=data_max,alpha=alpha,levels=levels)
                            for value in meta_plots[i*D_1 + j]:
                                grid[i*D_1 + j].scatter(value["x"],value["y"], c=value["color"], marker=value["marker"], s=80)#, label = value["label"])#
                            grid[i*D_1 + j].axis([d['X'].min(), d['X'].max(), d['Y'].min(), d['Y'].max()])
                        else:
                            if lognorm:
                                CS = grid[i*D_1 + j].pcolormesh(d['X'], d['Y'],zi,cmap=colour_map,vmin=data_min,vmax=data_max, alpha=alpha, norm=log_norm)#, levels=levels)
                                grid[i*D_1 + j].axis([d['X'].min(), d['X'].max(), d['Y'].min(), d['Y'].max()])
                            else:
                                CS = grid[i*D_1 + j].pcolormesh(d['X'], d['Y'], zi,cmap=colour_map,vmin=data_min,vmax=data_max,alpha=alpha)#,levels=levels)
                                grid[i*D_1 + j].axis([d['X'].min(), d['X'].max(), d['Y'].min(), d['Y'].max()])
                            if meta_plots: ## WAS UP
                                for value in meta_plots[i*D_1 + j]:
                                    grid[i*D_1 + j].scatter(value["x"],value["y"], c=value["color"], marker=value["marker"], s=80)#, label = value["label"])#
                        #grid[i*D_1 + j].xlim(4,53)
                        #grid[i*D_1 + j].ylim()
                        
                        dim_grid = d["dim_grid"]
                        extra =""
                        if (dim_grid[1] is None):
                            pass
                        else:
                            extra = str(int(dim_grid[1][j])) + "_"
                        im_title = extra + str(int(dim_grid[0][i]))
                        
                        t = MLOImageViewer.add_inner_title(grid[i*D_1 + j], im_title, loc=1)
                        counter = 0

                
                if meta_plots:
                    grid.cbar_axes[0].colorbar(CS,ticks=levels,alpha=alpha)
                    grid.cbar_axes[0].set_xticklabels([v for k, v in d['fitness'].error_labels.items()],
                                rotation='horizontal',
                                multialignment = 'right',
                                horizontalalignment = "left",
                                fontsize=16)
                else:
                    #for cbar_ax in grid.cbar_axes: 
                    #    cbar_ax.colorbar(CS)#,alpha=alpha)#,ticks=CS.levels)
                    #    cbar_ax.tick_params(labelsize=14)
                    grid.cbar_axes[0].colorbar(CS,alpha=alpha,format='%.1f')#,ticks=CS.levels)#,norm = LogNorm(vmax=data_max,vmin=data_min))#,ticks=CS.levels)
                    grid.cbar_axes[0].tick_params(labelsize=16)
                    
                    if False:
                        cmin,cmax = grid.cbar_axes[0].get_clim()
                        ticks = np.linspace(cmin,cmax,2)
                        grid.cbar_axes[0].set_ticks(ticks)
                        grid.cbar_axes[0].set_ticklabels(['%.1g' %2**t for t in ticks])
                    
                #fig.suptitle(title, fontsize=ImageViewer.TITLE_FONT_SIZE)
                #fig.suptitle(info, fontsize=ImageViewer.TITLE_FONT_SIZE)
                if True:
                    for ax in grid.axes_all: 
                        #ax.xaxis.set_minor_locator(FixedLocator([0,4,16,20,24,28],nbins=4))
                        ax.xaxis.set_major_locator(MaxNLocator(trim = False, nbins=5, steps=[0,1,2,3,4,7,15,16], prune=None, integer=True, symmetric=False))
                        ax.yaxis.set_major_locator(MaxNLocator(6))
                        ax.tick_params(direction='out', labelsize='16',which='major', length=4) 
                        
                        for xlabel in ax.get_xticklabels():
                            xlabel.set_rotation(-90)
                if False:
                    for ax in grid.axes_all:
                        ax.xaxis.set_major_locator(NullLocator())
                        ax.yaxis.set_major_locator(NullLocator())
                        
                #title.replace("Proximity Query Design Throughput Benchmark ", "") ## remove whitespaces
                fig.suptitle("Proximity Query Design Throughput Benchmark", fontsize=26)
                ImageViewer.save_fig(fig, str(d['images_folder']) + '/' + title + '_' + str(d['counter']) + '.png', ImageViewer.DPI)
            except TypeError,e:
                logging.error('Could not create ' + str(title) + ' plot for the GPR plot: ' + str(e))
    
    @staticmethod
    def save_fig(figure, filename, DPI):
        logging.info('Save fig ' + str(filename))
        #figure.subplots_adjust(left = (5/25.4)/figure.xsize, bottom = (4/25.4)/figure.ysize, right = 1 - (1/25.4)/figure.xsize, top = 1 - (3/25.4)/figure.ysize)#.tight_layout() 
        if ImageViewer.DIMS==2:
            figure.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
            figure.savefig(filename, dpi=DPI)#, bbox_inches='tight')
            figure.clf()
        elif ImageViewer.DIMS==3:
            figure.subplots_adjust(top=1.6)#left=0.01, right=0.99, top=0.99, bottom=0.01)
            figure.savefig(filename, dpi=DPI, bbox_inches='tight')
            figure.clf()
        elif ImageViewer.DIMS==4:
            figure.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
            figure.savefig(filename, dpi=DPI, bbox_inches='tight')
            figure.clf()

        
    #### COST FUNCTION PLOTS
                
    @staticmethod
    def plot_cost_function(figure, d):
        logging.debug("Plotting Cost...")
        fitness = d['fitness']
        try:            
            data = array([fitness.fitnessFunc(a, d['fitness_state'])[0][3][0] for a in d['z']])
        except:
            data = array([fitness.fitnessFunc(a)[3][0] for a in d['z']]) ###no fitness state
        graph_dict = d['all_graph_dicts']['Cost']
        ImageViewer.render_2d(figure, d, graph_dict, "Cost", data=data, fitness=fitness, minVal = fitness.cost_minVal, maxVal = fitness.cost_maxVal)
        
    @staticmethod
    def plot_cost_function_3_4d(figure, d):
        
        dim_grid = d["dim_grid"]
        fitness = d['fitness']
        
        D_0 = len(dim_grid[0])
        #logging.info(str(d['z']))
        dimension = len(d['fitness'].designSpace)
        ###we duplicate a lot of stuff cause of speed...
        if dimension == 4:
            D_1 = len(dim_grid[1])
            data = [[0.0] * D_1 for i in range(D_0)]
            mask = [[0.0] * D_1 for i in range(D_0)]
            for i in range(D_0):
                for j in range(D_1):
                    try: 
                        data[i][j] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i],dim_grid[1][j]]), d['fitness_state'])[0][3][0] for a in d['z']])
                        mask[i][j] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i],dim_grid[1][j]]), d['fitness_state'])[0][1][0] for a in d['z']])
                    except:
                        data[i][j] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i],dim_grid[1][j]]))[3][0] for a in d['z']]) ###no fitness state
                        mask[i][j] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i],dim_grid[1][j]]))[1][0] for a in d['z']]) ###no fitness state
        else:
            D_1 = 1
            data = [[0.0] * D_1 for i in range(D_0)]
            mask = [[0.0] * D_1 for i in range(D_0)]
            #logging.info("----" + str(D_0) + '   ' + str(D_1))
            #logging.info(str(data[0]))
            for i in range(D_0):
                try: 
                    data[i][0] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i]]), d['fitness_state'])[0][3][0] for a in d['z']])
                    mask[i][0] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i]]), d['fitness_state'])[0][1][0] for a in d['z']])
                except Exception,e:
                    data[i][0] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i]]))[3][0] for a in d['z']]) ###no fitness state
                    mask[i][0] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i]]))[1][0] for a in d['z']]) ###no fitness state
                    
        logging.info("Cost prepared")        
        graph_dict = d['all_graph_dicts']['Cost']
        ImageViewer.render_3_4d(figure, "Plotting Cost...", d, graph_dict, "Cost", data, lognorm=True, minVal = fitness.cost_minVal, maxVal = fitness.cost_maxVal)    

        
#### UTILITIES
        
    @staticmethod
    def figure_wrapper(figure, graph_dict, d, name, three_d=True):
                       
        if ImageViewer.SAVE_ALONE:
            my_figure = mpl.pyplot.figure()
            #my_figure = figure
            if three_d:
                plot = my_figure.add_subplot(111, projection='3d', elev=30)
            else:
                plot = my_figure.add_subplot(111)
            #plot.set_title(name, fontsize=ImageViewer.TITLE_FONT_SIZE)
            
            def save_plot():
                filename = str(d['images_folder']) + '/plot' + str(d['counter']) + "_" + name + '.png'
                if os.path.isfile(filename):
                    os.remove(filename)
                try:
                    ImageViewer.save_fig(my_figure, filename, ImageViewer.DPI)
                except:
                    logging.error('ImageViewer could not render a figure',exc_info=sys.exc_info())
                mpl.pyplot.close(my_figure)
            return plot, save_plot 
        else:
            if three_d:
                plot = figure.add_subplot(int(graph_dict['position']), projection='3d', elev=20)
            else:
                plot = figure.add_subplot(int(graph_dict['position']))
            #plot.set_title(name, fontsize=ImageViewer.TITLE_FONT_SIZE)
            def save_dummy(d, name):
                pass
            return plot, save_dummy
        
    @staticmethod
    def plot_MU_S2_EI_grid(figure, d, mask):
        
        dim_grid = d["dim_grid"]
        fitness = d['fitness']
        try:
            d['meta_plot']['Training_set'] = {'marker':"x",'color':"white",'data':d['classifier'].training_set}
        except:
            d['meta_plot'] = {}
            d['meta_plot']['Training_set'] = {'marker':"x",'color':"white",'data':d['classifier'].training_set}
        meta_plots = []
        
        D_0 = len(dim_grid[0])
        if not (dim_grid[1] is None):
            D_1 = len(dim_grid[1])
            data = [[[0.0] * D_1 for i in range(D_0)] for ii in range(5)]
            for i in range(D_0):
                for j in range(D_1):
                    #if d["propa_classifier"]:
                        #data = array([item[0] for item in EI]) * (reshape(d['classifier'].predict(d['z']),-1))
                    results = d['regressor'].predict([append(a,[dim_grid[0][i],dim_grid[1][j]]) for a in d['z']], with_EI=True)
                    for k in range(3):
                        data[k][i][j] = array([item[0] for item in results[k]])
                    try:
                        if d["propa_classifier"]:
                            meta_plotss = []
                            for key in d['meta_plot'].keys():
                                meta_plot = {}
                                meta_plot["color"] = d['meta_plot'][key]["color"]
                                meta_plot["marker"] = d['meta_plot'][key]["marker"]
                                meta_plot["label"] = key
                                meta_plot_data = d['meta_plot'][key]["data"]
                                meta_plot["x"] = array([item[0] for item in meta_plot_data if (item[2]-fitness.designSpace[2]["min"] ==i and item[3]-fitness.designSpace[3]["min"] ==j)])
                                meta_plot["y"] = array([item[1] for item in meta_plot_data if (item[2]-fitness.designSpace[2]["min"] ==i and item[3]-fitness.designSpace[3]["min"] ==j)])
                                meta_plotss.append(meta_plot)
                            meta_plots.append(meta_plotss)
                            data[3][i][j] = reshape(d['classifier'].predict([append(a,[dim_grid[0][i]]) for a in d['z']]),-1)
                            data[4][i][j] = data[3][i][j]
                    except:
                        pass
        else:
            D_1 = 1
            data = [[[0.0] * D_1 for i in range(D_0)] for ii in range(5)]
            for i in range(D_0):
                results = d['regressor'].predict([append(a,[dim_grid[0][i]]) for a in d['z']], with_EI=True)
                for k in range(3):
                    data[k][i][0] = array([item[0] for item in results[k]])  
                try:
                    if d["propa_classifier"]:
                        meta_plotss = []
                        for key in d['meta_plot'].keys():
                            meta_plot = {}
                            meta_plot["color"] = d['meta_plot'][key]["color"]
                            meta_plot["marker"] = d['meta_plot'][key]["marker"]
                            meta_plot["label"] = key
                            meta_plot_data = d['meta_plot'][key]["data"]
                            meta_plot["x"] = array([item[0] for item in meta_plot_data if item[2] - fitness.designSpace[2]["min"] ==i])
                            meta_plot["y"] = array([item[1] for item in meta_plot_data if item[2] - fitness.designSpace[2]["min"] ==i])
                            meta_plotss.append(meta_plot)
                        meta_plots.append(meta_plotss)
                        data[3][i][0] = reshape(d['classifier'].predict([append(a,[dim_grid[0][i]]) for a in d['z']]),-1)
                        data[4][i][0] = data[2][i][0] * data[3][i][0]
                except:
                    pass
        logging.debug("Prediction done passed...")
                    
        logging.info("Data for MU, S2, EI prepared")
        graph_dict = d['all_graph_dicts']['Fitness']
        ImageViewer.render_3_4d(figure, "MU", d, graph_dict, "$\hat{f}(\mathbf{x})$", data[0], fitness=d['fitness'],lognorm=True)#, mask=mask)
        ImageViewer.render_3_4d(figure, "S2", d, graph_dict, "$\sigma(\mathbf{x})$", data[1])
        ImageViewer.render_3_4d(figure, "EI", d, graph_dict, "EI", data[2],lognorm=True)
        try:
            if d["propa_classifier"]:
                ImageViewer.render_3_4d(figure, "PR_" + str(d['classifier'].best_gamma), d, graph_dict, "PROPA", data[3], meta_plots=meta_plots)
                ImageViewer.render_3_4d(figure, "dirtyEI", d, graph_dict, "dirtyEI", data[4],lognorm=True, meta_plots=meta_plots)
        except:
            pass
            
    @staticmethod
    def plot_MU_S2_EI(figure, d):
        logging.debug("Plotting Mean, S2, EI...")
        if not (d['regressor'] is None):  
            MU, S2, EI, P = d['regressor'].predict(d['z'])
            data = array([item[0] for item in MU])
            graph_dict = d['all_graph_dicts']['Mean']
            ImageViewer.render_2d(figure, d, graph_dict, "$\hat{f}(\mathbf{x})$", data=data, fitness=d['fitness'])
            try:
                if d["propa_classifier"]:
                    data = array([item[0] for item in EI]) * (reshape(d['classifier'].predict(d['z']),-1))
                else:
                    data = array([item[0] for item in EI])
            except:
                data = array([item[0] for item in EI])
            ImageViewer.render_2d(figure, d, graph_dict, "EI", data=data, fitness=d['fitness'], maxVal=data.max(),minVal=data.min())
            data = array([item[0] for item in S2])
            graph_dict = d['all_graph_dicts']['S2']
            ImageViewer.render_2d(figure, d, graph_dict, "$\sigma(\mathbf{x})$", data=data, fitness=d['fitness'], maxVal=data.max(),minVal=data.min())

    @staticmethod
    def plot_fitness_function_grid(figure, d):
        
        dim_grid = d["dim_grid"]
        fitness = d['fitness']
        
        D_0 = len(dim_grid[0])
        #logging.info(str(d['z']))
        dimension = len(d['fitness'].designSpace)
        ###we duplicate a lot of stuff cause of speed...
        if dimension == 4:
            D_1 = len(dim_grid[1])
            data = [[0.0] * D_1 for i in range(D_0)]
            mask = [[0.0] * D_1 for i in range(D_0)]
            for i in range(D_0):
                for j in range(D_1):
                    try: 
                        data[i][j] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i],dim_grid[1][j]]), d['fitness_state'])[0][0][0] for a in d['z']])
                        mask[i][j] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i],dim_grid[1][j]]), d['fitness_state'])[0][1][0] for a in d['z']])
                    except:
                        data[i][j] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i],dim_grid[1][j]]))[0][0] for a in d['z']]) ###no fitness state
                        mask[i][j] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i],dim_grid[1][j]]))[1][0] for a in d['z']]) ###no fitness state
        else:
            D_1 = 1
            data = [[0.0] * D_1 for i in range(D_0)]
            mask = [[0.0] * D_1 for i in range(D_0)]
            #logging.info("----" + str(D_0) + '   ' + str(D_1))
            #logging.info(str(data[0]))
            for i in range(D_0):
                try: 
                    data[i][0] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i]]), d['fitness_state'])[0][0][0] for a in d['z']])
                    mask[i][0] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i]]), d['fitness_state'])[0][1][0] for a in d['z']])
                except Exception,e:
                    data[i][0] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i]]))[0][0] for a in d['z']]) ###no fitness state
                    mask[i][0] = array([fitness.fitnessFunc(append(a,[dim_grid[0][i]]))[1][0] for a in d['z']]) ###no fitness state
                    
        logging.info("Fitness prepared")        
        graph_dict = d['all_graph_dicts']['Fitness']
        ImageViewer.render_3_4d(figure, "Plotting fitness function grid", d, graph_dict, "$f(\mathbf{x})$", data,lognorm=True, mask=mask, fitness=d['fitness'])    
                
    @staticmethod
    def plot_fitness_function(figure, d):
        logging.debug("Plotting Fitness...")
        fitness = d['fitness']
        try:            
            data = array([fitness.fitnessFunc(a, d['fitness_state'])[0][0][0] for a in d['z']])
        except:
            data = array([fitness.fitnessFunc(a)[0][0] for a in d['z']]) ###no fitness state
        graph_dict = d['all_graph_dicts']['Fitness']
        ImageViewer.render_2d(figure, d, graph_dict, "$f(\mathbf{x})$", data=data, fitness=fitness)

    @staticmethod
    def plot_fitness_progression(figure, d):
        logging.info("Plotting Fitness Progression...")
        graph_dict = d['all_graph_dicts']['Progression']
        plot = figure.add_subplot(int(graph_dict['position']))

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=ImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
        plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)

        ### Other settings
        try:
            plot.set_xlim(1,   max(10, max(d['generations_array'])))
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass
        try:
            plot.set_ylim(0.0, max(d['best_fitness_array']) * 1.1)
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass 
           
        ### Data
        plot.plot(d['generations_array'], d['best_fitness_array'],
                  c='red', marker='x')

## This class containts 
class MLOImageViewer(ImageViewer):
    DPI = 300
    SAVE_ALONE = True

    LABEL_FONT_SIZE = 10
    TITLE_FONT_SIZE = 26

    @staticmethod
    def render(input_dictionary):
        logging.info("Rendering...")
        if input_dictionary["generate"]:
            dictionary = MLOImageViewer.get_default_attributes() ##this way the default view will be used if different one was not supplied
            figure = mpl.pyplot.figure(figsize=(24,12))
            dictionary.update(input_dictionary)
            figure.subplots_adjust(wspace=0.15, hspace=0.35)
            counter_headers = []
            header = []
            figure.suptitle(dictionary['graph_title'])

            rerender = True ## pointless... remove
            designSpace = dictionary['fitness'].designSpace
            npts = 100

            ### Initialize some graph points
            dimensions = len(designSpace)
            ImageViewer.DIMS = dimensions
            if dimensions == 2 :
                x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
                y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
                x, y = meshgrid(x, y)
                dictionary['x'] = reshape(x, -1)
                dictionary['y'] = reshape(y, -1)
                dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],
                                                                  dictionary['y'])])

                dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                            designSpace[0]['max'] + 0.01, npts)
                dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                            designSpace[1]['max'] + 0.01, npts)
                    
                                            
                dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],
                                                            dictionary['yi'])
                                                            
                                                            
                if dictionary['all_graph_dicts']['DesignSpace']['generate']:
                    MLOImageViewer.plot_design_space(figure, dictionary)
                if dictionary['all_graph_dicts']['Mean']['generate'] or dictionary['all_graph_dicts']['EI']['generate'] or dictionary['all_graph_dicts']['S2']['generate']:
                    ImageViewer.plot_MU_S2_EI(figure, dictionary)
                if True:#dictionary['all_graph_dicts']['Fitness']['generate']:
                    ImageViewer.plot_fitness_function(figure, dictionary)
                if False:#dictionary['all_graph_dicts']['Cost']['generate']:
                    ImageViewer.plot_cost_function(figure, dictionary)
                if False:#dictionary['all_graph_dicts']['Progression']['generate']:
                    ImageViewer.plot_fitness_progression(figure, dictionary)
                ### Save and exit
                filename = str(dictionary['images_folder']) + '/plot' + str(dictionary['counter']) + '.png'
                if rerender and os.path.isfile(filename):
                    os.remove(filename)
                try:
                    #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                    #                                             Plot_View.DPI))
                    MLOImageViewer.save_fig(figure, filename, MLOImageViewer.DPI)
                except:
                    logging.error(
                        'MLOImageViewer could not render a plot',exc_info=sys.exc_info())
                figure.tight_layout()
                mpl.pyplot.close(figure)
            elif (dimensions == 3) or (dimensions == 4):
                ## we define a grid 
                
                if designSpace[0]['type'] == "discrete":
                    x = arange(designSpace[0]['min'], designSpace[0]['max'] + designSpace[0]['step'], designSpace[0]['step'])
                else:
                    x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
                    
                if designSpace[1]['type'] == "discrete":
                    y = arange(designSpace[1]['min'], designSpace[1]['max'] + designSpace[1]['step'], designSpace[1]['step'])
                else:
                    y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
                dim_grid =[None,None]
                if designSpace[2]['type'] == "discrete":
                    dim_grid[0] = arange(designSpace[2]['min'], designSpace[2]['max'] + designSpace[2]['step'], designSpace[2]['step'])
                else: # continous
                    dim_grid[0] = linspace(designSpace[2]['min'], designSpace[2]['max'], npts)
                    
                if dimensions == 4:
                    if  designSpace[3]['type'] == "discrete":
                        dim_grid[1] = arange(designSpace[3]['min'], designSpace[3]['max'] + designSpace[3]['step'], designSpace[3]['step'])
                    else:
                        dim_grid[1] = linspace(designSpace[3]['min'], designSpace[3]['max'], npts)
                dictionary["dim_grid"] = dim_grid
                x, y = meshgrid(x, y)
                dictionary['x'] = reshape(x, -1)
                dictionary['y'] = reshape(y, -1)
                dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],
                                                                  dictionary['y'])])

                ### Define grid
                dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                            designSpace[0]['max'] + 0.01, npts)
                dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                            designSpace[1]['max'] + 0.01, npts)
                
                                            
                dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],
                                                            dictionary['yi'])
            
                if True:#dictionary['all_graph_dicts']['Fitness']['generate']:
                    MLOImageViewer.plot_fitness_function_grid(figure, dictionary)
                if False:#dictionary['all_graph_dicts']['DesignSpace']['generate']:
                    mask = MLOImageViewer.plot_design_space_grid(figure, dictionary)
                if False:
                    MLOImageViewer.plot_MU_S2_EI_grid(figure, dictionary, mask=mask)
                if False:
                    MLOImageViewer.plot_cost_function_3_4d(figure, dictionary)
                ## ADD COST
                
            else:
                logging.info("We only support visualization of 2, 3 and 4 dimensional spaces")
            
            #sys.exit(0) ## I let it as a reminder... do NOT uncomment this! will get the applciation to get stuck
        else: ## do not regenerate
            pass
        
    #### DESIGN SPACE PLOTS
    @staticmethod
    def plot_design_space(figure, d):
        logging.info("Plotting Design Space...")
        graph_dict = d['all_graph_dicts']['DesignSpace']
        plot, save_fig  = MLOImageViewer.figure_wrapper(figure, graph_dict, d, "DesignSpace", three_d=False)
        fitness = d['fitness']
        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(fitness.get_x_axis_name(), fontsize=font_size+6)
        plot.set_ylabel(fitness.get_y_axis_name(), fontsize=font_size+6)
        colour_map = mpl.cm.get_cmap(graph_dict['colour map'])
        xcolour = graph_dict['x-colour']
        ocolour = graph_dict['o-colour']

        ### Other settings
        locator = LinearLocator(4)
        locator.tick_values(fitness.designSpace[0]["min"], fitness.designSpace[0]["max"])
        plot.xaxis.set_major_locator(locator)
        
        locator = MaxNLocator(5,integer=True)
        locator.tick_values(fitness.designSpace[1]["min"], fitness.designSpace[1]["max"])
        plot.yaxis.set_major_locator(locator)
        
        plot.tick_params(axis='both', which='major', labelsize=20)
        ### Data
        fitness = d['fitness']
        if not (d['classifier'] is None):
            zClass = d['classifier'].predict(d['z'])
            
            zi3 = griddata((d['x'], d['y']), zClass, (d['xi'][None, :], d['yi'][:, None]), method='nearest')

            levels = [k for k, v in fitness.error_labels.items()]
            levels = [l-0.1 for l in levels]
            levels.append(levels[-1]+1.0)
            CS = plot.contourf(d['X'], d['Y'], zi3, levels, cmap=colour_map,  alpha = 0.7)
            
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(mpl.pyplot.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = mpl.pyplot.colorbar(CS, cax=cax, ticks=CS.levels)
            cbar.ax.set_yticklabels([ "                  " + v #+ " " * (int(len(v)/3))
                                     for k, v in fitness.error_labels.items()],
                                    rotation='vertical',
                                    fontsize=font_size)
            
            #
            plot_trainingset_x = [] 
            plot_trainingset_y = []
            training_set = d['classifier'].training_set
            training_labels = d['classifier'].training_labels
            
            for i in range(0, len(training_set)):
                p = training_set[i]
                plot_trainingset_x.append(p[0])
                plot_trainingset_y.append(p[1])
        
            ## plot meta-heuristic specific markers 
            ## TODO - come up with a way of adding extra colours
            #print d['all_graph_dicts']
            for key in d['meta_plot'].keys():
                data = d['meta_plot'][key]["data"]
                x_0 = array([item[0] for item in data])
                x_1 = array([item[1] for item in data])
                plot.scatter(x_0,x_1, c="white",marker=d['meta_plot'][key]["marker"], s=150, label = "Particles")

            if len(plot_trainingset_x) > 0:
                plot.scatter(x=plot_trainingset_x, y=plot_trainingset_y, c="black", marker='x', s=150, label = "Evaluations")
            plot.axis([d['X'].min(), d['X'].max(), d['Y'].min(), d['Y'].max()])
            try:
                logging.info("Transforming Axis Labels")
                x_func, y_func = MLOImageViewer.my_formatter(fitness)
                plot.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(x_func))
                plot.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(y_func))
                def two_digits(x, pos):
                    return '%1.2f' % x
            except Exception,e:
                logging.info("KRUWAAA " + str(e))
                pass
             
            plot.legend( loc='upper left', numpoints = 1)
            save_fig()
    
    @staticmethod
    def plot_design_space_grid(figure, d):
        dim_grid = d["dim_grid"]
        fitness = d['fitness']
        D_0 = len(dim_grid[0])
        ###we duplicate a lot of stuff cause of speed...
        meta_plots = []
        try:
            d['meta_plot']['Training_set'] = {'marker':"x",'color':"black",'data':d['classifier'].training_set}
        except:
            d['meta_plot'] = {}
            d['meta_plot']['Training_set'] = {'marker':"x",'color':"black",'data':d['classifier'].training_set}
            
        if not (dim_grid[1] is None):
            D_1 = len(dim_grid[1])
            data = [[0.0] * D_1 for i in range(D_0)]
            for i in range(D_0):
                for j in range(D_1):
                    meta_plotss = []
                    for key in d['meta_plot'].keys():
                        meta_plot = {}
                        meta_plot["color"] = d['meta_plot'][key]["color"]
                        meta_plot["marker"] = d['meta_plot'][key]["marker"]
                        meta_plot["label"] = key
                        meta_plot_data = d['meta_plot'][key]["data"]
                        meta_plot["x"] = array([item[0] for item in meta_plot_data if (item[2]==i and item[3] ==j)])
                        meta_plot["y"] = array([item[1] for item in meta_plot_data if (item[2]==i and item[3] ==j)])
                        meta_plotss.append(meta_plot)
                    meta_plots.append(meta_plotss)
                    data[i][j] = d['classifier'].predict([append(a,[dim_grid[0][i],dim_grid[1][j]]) for a in d['z']])
                    
        else:
            D_1 = 1
            data = [[0.0] * D_1 for i in range(D_0)]
            for i in range(D_0):
                meta_plotss = []
                for key in d['meta_plot'].keys():
                    meta_plot = {}
                    meta_plot["color"] = d['meta_plot'][key]["color"]
                    meta_plot["marker"] = d['meta_plot'][key]["marker"]
                    meta_plot["label"] = key
                    meta_plot_data = d['meta_plot'][key]["data"]
                    meta_plot["x"] = array([item[0] for item in meta_plot_data if item[2]==i])
                    meta_plot["y"] = array([item[1] for item in meta_plot_data if item[2]==i])
                    meta_plotss.append(meta_plot)
                meta_plots.append(meta_plotss)
                data[i][0] = d['classifier'].predict([append(a,[dim_grid[0][i]]) for a in d['z']])
                
                
        levels = [k for k, v in fitness.error_labels.items()]
        levels = [l-0.1 for l in levels]
        levels.append(levels[-1]+1.0)
        logging.info(levels)
        
        graph_dict = d['all_graph_dicts']['DesignSpace']
        #logging.info(str(data))
        logging.info("Design Space prepared")       
        MLOImageViewer.render_3_4d(figure, "design_space", d, graph_dict, "Design Space", data, meta_plots=meta_plots, levels = levels, cmap_string = "gist_yarg", alpha=0.5)
        return data
        
    @staticmethod
    def get_attributes(name):
    
        attribute_dictionary = {
            'All':         ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'x-colour', 'o-colour',
                            'position'],
            'DesignSpace':         ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'colour map', 'x-colour', 'o-colour', 'position'],
            'Mean':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Progression': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'position'],
            'Fitness':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Cost':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position']
        }
        return attribute_dictionary.get(name, None)
        
    @staticmethod
    def get_default_attributes():
        # Default values for describing graph visualization
        graph_title = 'Title'
        graph_names = ['Progression', 'Fitness', 'Mean', 'DesignSpace', "Cost", 'EI', 'S2', ]

        graph_dict1 = {'subtitle': 'Currently Best Found Solution',
                       'x-axis': 'Iteration',
                       'y-axis': 'Fitness',
                       'font size': '22',
                       'position': '241'}
        graph_dict2 = {'subtitle': 'Fitness Function',
                       'x-axis': '$p$',
                       'y-axis': '$freq$',
                       'z-axis': 'Execution Time',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '242'}
        graph_dict3 = {'subtitle': 'Regression Mean',
                       'x-axis': '$p$',
                       'y-axis': '$freq$',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '243'}
        graph_dict4 = {'subtitle': 'Design Space',
                       'x-axis': '$p$',
                       'y-axis': '$freq$',
                       'font size': '22',
                       'colour map': 'gray',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '244'}
        graph_dict5 = {'subtitle': 'Cost Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '22',
                       'colour map': 'gray',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '245'}
        graph_dict6 = {'subtitle': 'Regression EI',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '246'}
        graph_dict7 = {'subtitle': 'Regression S2',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '247'}
        all_graph_dicts = {'Progression': graph_dict1,
                           'Fitness': graph_dict2,
                           'Mean': graph_dict3,
                           'DesignSpace': graph_dict4,
                           'Cost': graph_dict5,
                           'EI': graph_dict6,
                           'S2': graph_dict7
                           }
                           
        
            
        graph_dictionary = {
            'rerendering': False,
            'graph_title': graph_title,
            'graph_names': graph_names,
            'all_graph_dicts': all_graph_dicts
        }
        
        for name in graph_names:
            graph_dictionary['all_graph_dicts'][name]['generate'] = True
        graph_dictionary['all_graph_dicts']['S2']['generate'] = False                  
        graph_dictionary['all_graph_dicts']['Progression']['generate'] = False                  
        return graph_dictionary
        
## This class returns a pdf containing a summary of the runs
## 
class MLORunReportViewer2(ImageViewer):

    DPI = 400
    LABEL_FONT_SIZE = 10
    TITLE_FONT_SIZE = 10

    @staticmethod
    def render(input_dictionary):
        if input_dictionary["generate"]:
            dictionary = MLOImageViewer.get_default_attributes() ##this way the default view will be used if different one was not supplied
            dictionary.update(input_dictionary)
            figure = mpl.pyplot.figure()
            figure.subplots_adjust(wspace=0.35, hspace=0.35)
            figure.suptitle(dictionary['graph_title'])

            rerender = True ## pointless... remove
            designSpace = dictionary['fitness'].designSpace
            npts = 100

            ### Initialize some graph points
            x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
            y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
            x, y = meshgrid(x, y)
            dictionary['x'] = reshape(x, -1)
            dictionary['y'] = reshape(y, -1)
            dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],
                                                              dictionary['y'])])

            ### Define grid
            dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                        designSpace[0]['max'] + 0.01, npts)
            dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                        designSpace[1]['max'] + 0.01, npts)
            dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],
                                                        dictionary['yi'])

            ### Generate the graphs according to the user's selection
            if dictionary['all_graph_dicts']['Mean']['generate']:
                MLOImageViewer.plot_MU(figure, dictionary)
            if dictionary['all_graph_dicts']['Fitness']['generate']:
                MLOImageViewer.plot_fitness_function(figure, dictionary)
            if dictionary['all_graph_dicts']['Progression']['generate']:
                MLOImageViewer.plot_fitness_progression(figure, dictionary)
            if dictionary['all_graph_dicts']['DesignSpace']['generate']:
                MLOImageViewer.plot_design_space(figure, dictionary)

            ### Save and exit
            filename = str(dictionary['images_folder']) + '/plot' + str(dictionary['counter']) + '.png'
            if rerender and os.path.isfile(filename):
                os.remove(filename)
            try:
                #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                #                                             Plot_View.DPI))
                MLOImageViewer.save_fig(figure, filename, MLOImageViewer.DPI)
            except:
                logging.error(
                    'MLOImageViewer could not render a plot for ' + str(name),
                    exc_info=sys.exc_info())
            mpl.pyplot.close(figure)
            #sys.exit(0) ## I let it as a reminder... do NOT uncomment this! will get the applciation to get stuck
        else: ## do not regenerate
            pass
        
    @staticmethod
    def save_fig(figure, filename, Format):
        logging.info('Save fig ' + str(filename))
        figure.savefig(filename, dpi=DPI)
        
    @staticmethod
    def get_attributes(name):
    
        attribute_dictionary = {
            'All':         ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'x-colour', 'o-colour',
                            'position'],
            'DesignSpace':         ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'colour map', 'x-colour', 'o-colour', 'position'],
            'Mean':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Progression': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'position'],
            'Fitness':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position']
        }
        return attribute_dictionary.get(name, None)
        
    @staticmethod
    def get_default_attributes():
        # Default values for describing graph visualization
        graph_title = 'Title'
        graph_names = ['Progression', 'Fitness', 'Mean', 'DesignSpace']

        graph_dict1 = {'subtitle': 'Currently Best Found Solution',
                       'x-axis': 'Iteration',
                       'y-axis': 'Fitness',
                       'font size': '22',
                       'position': '221'}
        graph_dict2 = {'subtitle': 'Fitness Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '222'}
        graph_dict3 = {'subtitle': 'Regression Mean',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '223'}
        graph_dict4 = {'subtitle': 'Design Space',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'font size': '22',
                       'colour map': 'gray',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '224'}
        all_graph_dicts = {'Progression': graph_dict1,
                           'Fitness': graph_dict2,
                           'Mean': graph_dict3,
                           'DesignSpace': graph_dict4}
                           
        
            
        graph_dictionary = {
            'rerendering': False,
            'graph_title': graph_title,
            'graph_names': graph_names,
            'all_graph_dicts': all_graph_dicts
        }
        
        for name in graph_names:
            graph_dictionary['all_graph_dicts'][name]['generate'] = True
                          
        return graph_dictionary
        
##
class MLORunReportViewer(object):

    @staticmethod
    def render(dictionary):
        logging.info("Generating Report")
        ## Generate Header   
        header = ['Trial Name', 'Trial Number']  
        counter_headers = []
        timer_headers = []
        trial_snapshots = dictionary["trials_snapshots"]
        first_trial_snapshot = trial_snapshots[0]
        ## get counter names
        counters = first_trial_snapshot['counter_dict'].keys()
        for counter in counters: ## list of names of Counters
            header.append('Counter "' + counter + '"')
            counter_headers.append(counter)
        ## get timing names
        timers = first_trial_snapshot['timer_dict'].keys()
        for timer in timers: ## list of names of Counters
            header.append(timer)
            timer_headers.append(timer)
            
        htmlcode = HTML.Table(header_row=header)
        
        statistics = ["mean","std","max","min",]
        data = []
        
        for trial_snapshot in trial_snapshots:
            ## Display trial timers
            trial_name = [trial_snapshot["name"]]
            trial_no = [1]
            trial_counters = [trial_snapshot['counter_dict'][counter_header] for counter_header in counter_headers] 
            trial_timers = [trial_snapshot['timer_dict'][timer_header] for timer_header in timer_headers] 
            data.append(trial_counters + trial_timers)
            row = [HTML.TableCell(cell, bgcolor='Lime') for cell in trial_name + trial_no + trial_counters + trial_timers]
            htmlcode.rows.append(row)
            ## Display trial counters
        htmlcode = str(htmlcode)    
        
        data = array(data)
        htmlcode2 = HTML.Table(header_row=header)
        statistic_no = 0
        for statistic in statistics:
            statistic_name = statistic
            result = [str(elem) for elem in eval("data." + statistic + "(axis=0)")]
            row = [HTML.TableCell(cell, bgcolor='Lime') for cell in [statistic_name] + [str(statistic_no)] + result]
            statistic_no = statistic_no + 1
            htmlcode2.rows.append(row)
            ## append to list used to calculate statistical data
        htmlcode2 = str(htmlcode2)
        ### Save and exit
        filename = dictionary["results_folder_path"] + "/run_report.pdf"
        filename2 = dictionary["results_folder_path"] + "/run_report.html"
        
        if os.path.isfile(filename):
            os.remove(filename)
        try:
            f = file(filename, 'wb')
            pdf = pisa.CreatePDF(htmlcode + htmlcode2,f)
            if not pdf.err:
                pisa.startViewer(f)
            f.close()
        except Exception, e:
            logging.error('could not create a report for ' + str(e))
            
        if os.path.isfile(filename2):
            os.remove(filename2)
        try:
            f = file(filename2, 'wb')
            f.write(htmlcode + htmlcode2)
            f.close()
        except Exception, e:
            logging.error('could not create a report for ' + str(e))
        logging.info("Done Generating Report")

    @staticmethod
    def get_attributes(name):
        return {}
        
    @staticmethod
    def get_default_attributes():
        return {}

##This class returns a string 
##It should return either a string, a file reference or 
class MLORegressionReportViewer(object):
    @staticmethod
    def get_error_code(compare_dict):
            ## Initialize all run to be true
            color='lime'
            ErrorCode = []
            message = []
           
            # read from compare_dict
            trial_snapshot = compare_dict['trial_snapshot']
            trial_counters = compare_dict['trial_counters']
            trial_timers = compare_dict['trial_timers']
            golden_counters = compare_dict['goldenResult']['golden_counters']
            golden_timers = compare_dict['goldenResult']['golden_timers']
            counter_headers = compare_dict['goldenResult']['counter_headers']
            timer_headers = compare_dict['goldenResult']['timer_headers']
            
            ## Tell if one fails and give the errorCode and message
            if trial_snapshot['counter_dict']['fit']>trial_snapshot['max_fitness']:
                color = 'red'
                ErrorCode.append('1')
                message.append('Run out of fitness budget')
           
            if trial_snapshot['counter_dict']['g']>trial_snapshot['max_iter']: 
                color = 'red'
                ErrorCode.append('2')
                message.append('Run out of iteration budget')
            
            for i, (Counter,gCounter,Timer,gTimer,counter_header,timer_header) in enumerate(zip(trial_counters,golden_counters,trial_timers, golden_timers,counter_headers,timer_headers)):
                # compare the counters
                if int(Counter) > int(gCounter):
                    color = 'red'
                    if not '3' in ErrorCode:
                        ErrorCode.append('3')
                    outnumber = 100 * (int(Counter) - int(gCounter))/int(gCounter)
                    message.append(trial_snapshot["name"]+" -- The counter " + str(counter_header) + " outnumbers the golden result by " + str(outnumber) +"%.")
                    trial_counters[i] =str(trial_counters[i]) + ' ('+ str(outnumber) + '%)'
                # compare the timers
                if int(Timer) > int(gTimer):
                    color = 'red'
                    if not '4' in ErrorCode:
                        ErrorCode.append('4')
                    outnumber = 100 * (int(Timer) - int(gTimer))/int(gTimer)
                    message.append(trial_snapshot["name"]+" -- The timer " + str(timer_header) + " outnumbers the golden result by " + str(outnumber) +"%.")
                    trial_timers[i] =str(trial_timers[i]) + ' ('+ str(outnumber) + '%)'
            
            #output the errorCode
            ErrorC = ''
            if len(ErrorCode)==0:
                ErrorC = ErrorC + '0'
            else:
                for e in ErrorCode:
                    ErrorC = ErrorC + e + ' '
            
            
            return_dictionary = {
                                 'color': color,
                                 'ErrorCode' : [ErrorC],
                                 'trial_counters' : trial_counters,
                                 'trial_timers' : trial_timers,
                                 'message' : message
                                 }
            return return_dictionary
    
    @staticmethod
    def render(dictionary):
        logging.info("Generating Report...")
        ## Generate Header   
        header = ['Trial Name', 'Trial Number']  
        counter_headers = []
        timer_headers = []
        trial_snapshots = dictionary["trials_snapshots"]
        first_trial_snapshot = trial_snapshots[0]
        
        run_name = str(first_trial_snapshot['run_name'])
        #run_name = run_name[0:run_name.find('_')]
        ## get counter names
        counters = first_trial_snapshot['counter_dict'].keys()
        for counter in counters: ## list of names of Counters
            header.append('Counter "' + counter + '"')
            counter_headers.append(counter)
        ## get timing names
        timers = first_trial_snapshot['timer_dict'].keys()
        for timer in timers: ## list of names of Counters
            header.append(timer)
            timer_headers.append(timer)
        
        header.append('Error Code')
        htmlcode1 = HTML.Table(header_row=header)
        data = []
        failurecount=0
        failure_trial=[]
        
        # set golden file path
        goldenResultsFile = first_trial_snapshot['configuration_folder_path']+'goldenResult.txt'
        goldenResult = {}

        trial_message = []
        for trial_snapshot in trial_snapshots:
            ## Display trial timers
            trial_name = [trial_snapshot["name"]]
            trial_no = [1]
            trial_counters = [trial_snapshot['counter_dict'][counter_header] for counter_header in counter_headers] 
            trial_timers = [trial_snapshot['timer_dict'][timer_header] for timer_header in timer_headers] 
            data.append(trial_counters + trial_timers)
            
            # create golden file or read from the file
            if os.path.exists(goldenResultsFile):
                #read from file
                logging.info("Reading golden file...")
                with open(goldenResultsFile, 'rb') as outfile:
                    dict = pickle.load(outfile)
                    dict['dir'] = goldenResultsFile
                    goldenResult = dict
            else:
                #create new golden resultfile
                logging.info("A new golden result to be created")
                goldenResult = {
                               'Saved_Time': strftime("%d/%b/%Y %H:%M:%S", gmtime()),
                               'golden_counters' : trial_counters,
                               'golden_timers' : trial_timers,
                               'counter_headers' : counter_headers,
                               'timer_headers' : timer_headers,
                               'dir':goldenResultsFile
                               }
                with io.open(goldenResult['dir'], 'wb') as outfile:
                    pickle.dump(goldenResult, outfile)            

            compare_dict = {
                            'trial_snapshot' : trial_snapshot,
                            'trial_counters': trial_counters, 
                            'trial_timers' : trial_timers, 
                            'goldenResult' : goldenResult
                            }
            return_dictionary = MLORegressionReportViewer.get_error_code(compare_dict)
            color = return_dictionary['color']
            ErrorCode = return_dictionary['ErrorCode']
            trial_counters = return_dictionary['trial_counters']
            trial_timers = return_dictionary['trial_timers']
            trial_message.append(return_dictionary['message'])
            
            if color == 'red':
                failure_trial.append(trial_name)
                failurecount = failurecount + 1
            row = [HTML.TableCell( cell, bgcolor = color) for cell in trial_name + trial_no + trial_counters + trial_timers + ErrorCode]
            
            htmlcode1.rows.append( row )
            
        repo_message = ""
        for m in trial_message:
            if len(m)>0:
                for ms in m:
                    repo_message = repo_message + ms + '<br>'
        ## statistics
        header = ['Statistics', 'Total Trials']  
        ## get counter names
        counters = first_trial_snapshot['counter_dict'].keys()
        for counter in counters: ## list of names of Counters
            header.append('Counter "' + counter + '"')
        ## get timing names
        timers = first_trial_snapshot['timer_dict'].keys()
        for timer in timers: ## list of names of Counters
            header.append(timer)
            
        
        statistics = ["mean","std","max","min"]
        data = array(data)
        htmlcode2 = HTML.Table(header_row=header)
        total_trials=len(trial_snapshots)

        for statistic in statistics:
            statistic_name = statistic
            result = [str(elem) for elem in eval("data." + statistic + "(axis=0)")]
            row = [HTML.TableCell(cell) for cell in [statistic_name] + [str(total_trials)] + result]
            htmlcode2.rows.append(row)
            ## append to list used to calculate statistical data

        # information list in report
        headlist = []
        time=strftime("<hr><hgroup> <h3>Repo Time: %d/%b/%Y %H:%M:%S</h3></hgroup>", gmtime())
        headlist.append(time)
        
        # generate git information
        repo = git.Repo( os.getcwd() )
        headcommit = repo.head.commit
        headlist.append("Git Committer :  " + str(headcommit.committer))
        headlist.append("Commit Date : " + asctime(gmtime(headcommit.committed_date)))

        # generate run information
        headlist.append("Run Name: "+ run_name)
        
        # generate the regressor name and classifier name
        reg = str(trial_snapshots[0]['regressor'])
        cla = str(trial_snapshots[0]['classifier'])
        headlist.append("Regressor:     " + reg[34:reg.find("object")])
        headlist.append("Classifier:     " + cla[35:cla.find("object")])
        
        ## generate the trial information: fail or not?
        headlist.append( "Total Trials :  " + str(len(trial_snapshots)) )
        headlist.append( "Total Fails : " + str(failurecount))
        if failurecount>0:
            headlist.append("Fail trials: " + str(failure_trial))

        html_list = HTML.list(headlist)
        # create the report contents
        htmlcode1 = str( htmlcode1 )
        htmlcode2 = str( htmlcode2 )
        htmlcode3 = str ( html_list )
        repocontent = htmlcode3 + '<center> <b><font size="5">Trial information:</font></b> '+'<br>'+'[( ) shows how many percent it exceed the golden results]'+'<br>' + htmlcode1 +'<b>Regression Message from trials: </b>'+'<br>'+ repo_message +'</center><br>' + '<center><b><font size="5">Statistics:</font></b> ' + htmlcode2 + '</center><br>'
        
            
        ### Save and exit
        filename1 = dictionary["results_folder_path"] + "/run_report.pdf"
        filename2 = dictionary["results_folder_path"] + "/run_report.html"
        filename3 = first_trial_snapshot["run_folders_path"] + "/regression_report.html"
        
        # save pdf report 
        if os.path.isfile(filename1):
            os.remove(filename1)
        try:
            f = file(filename1, 'w')
            pdf = pisa.CreatePDF(repocontent,f)
            if not pdf.err:
                pisa.startViewer(f)
                logging.info("Viewing Report...")
                
            f.close()
        except Exception, e:
            logging.error('could not create a report for {}'.format(str(e)))
            
        # save html report 
        if os.path.isfile(filename2):
            os.remove(filename2)
        try:
            f = file(filename2, 'w')
            f.writelines(repocontent)
            f.close()

        except Exception, e:
            logging.error('could not create a report for {}'.format(str(e)))

        # save the report for all the runs in the folder
        loglist=os.listdir(first_trial_snapshot["run_folders_path"])
        reportfile=[]
        for i in range(0,len(loglist)):
            reportfile.append(first_trial_snapshot["run_folders_path"]+"/"+loglist[i]+"/run_report.html")
        f = file(filename3, 'w' )
        f.write("<hgroup> <center> <h2> MLO Report </h2> </center> </hgroup>")
        f.write("<center> (Error Code:  0 - No Error; 1 - Exceed Max Fitness, 2 - Exceed Max Generation, 3- Exceed the golden counters, 4 - Exceed the golden timers) </center>")
        for e in reportfile:
            if os.path.exists(e):
                fr = file(e,'r')
                temp = fr.readlines()
                f.writelines(temp)
                fr.close()
        f.close()

    @staticmethod
    def get_attributes(name):
        return {}
        
    @staticmethod
    def get_default_attributes():
        return {}
        
## This class containts 
class MLOTimeAware_ImageViewer(MLOImageViewer):

    DPI = 150
    LABEL_FONT_SIZE = 16
    TITLE_FONT_SIZE = 16

    @staticmethod
    def render(input_dictionary):
        if input_dictionary["generate"]:
            dictionary = MLOTimeAware_ImageViewer.get_default_attributes() ##this way the default view will be used if different one was not supplied
            dictionary.update(input_dictionary)
            figure = mpl.pyplot.figure()
            figure.subplots_adjust(wspace=0.35, hspace=0.35)
            figure.suptitle(dictionary['graph_title'])

            rerender = True ## pointless... remove
            designSpace = dictionary['fitness'].designSpace
            npts = 100

            ### Initialize some graph points
            logging.info(  "designSpace " + designSpace)
            x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
            y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
            x, y = meshgrid(x, y)
            
            dictionary['x'] = reshape(x, -1)
            
            dictionary['y'] = reshape(y, -1)
            
            dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],
                                                              dictionary['y'])])
            logging.info(  "z " + dictionary['z'])
            ### Define grid
            dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                        designSpace[0]['max'] + 0.01, npts)
            dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                        designSpace[1]['max'] + 0.01, npts)
            dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],
                                                        dictionary['yi'])

            ### Generate the graphs according to the user's selection
            if dictionary['all_graph_dicts']['Mean']['generate']:
                MLOImageViewer.plot_MU(figure, dictionary)
            if dictionary['all_graph_dicts']['Fitness']['generate']:
                MLOImageViewer.plot_fitness_function(figure, dictionary)
            if dictionary['all_graph_dicts']['Progression']['generate']:
                MLOImageViewer.plot_fitness_progression(figure, dictionary)
            if dictionary['all_graph_dicts']['DesignSpace']['generate']:
                MLOImageViewer.plot_design_space(figure, dictionary)
            if dictionary['all_graph_dicts']['Cost']['generate']:
                MLOImageViewer.plot_cost_function(figure, dictionary)
            if dictionary['all_graph_dicts']['Cost_model']['generate']:
                MLOTimeAware_ImageViewer.plot_cost_space_model(figure, dictionary)    
            ### Save and exit
            filename = str(dictionary['images_folder']) + '/plot' + str(dictionary['counter']) + '.png'
            if rerender and os.path.isfile(filename):
                os.remove(filename)
            try:
                #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                #                                             Plot_View.DPI))
                MLOTimeAware_ImageViewer.save_fig(figure, filename, MLOTimeAware_ImageViewer.DPI)
            except:
                logging.error(
                    'MLOTimeAware_ImageViewer could not render a plot for ' + str(name),
                    exc_info=sys.exc_info())
            mpl.pyplot.close(figure)
            #sys.exit(0) ## I let it as a reminder... do NOT uncomment this! will get the applciation to get stuck
        else: ## do not regenerate
            pass
        
    def plot_cost_space_model(figure, d):
        graph_dict = d['all_graph_dicts']['Cost_model']
        plot = figure.add_subplot(int(graph_dict['position']),
                                  projection='3d', elev=20)

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOTimeAware_ImageViewer.TITLE_FONT_SIZE)
        plot.set_ylabel('\n' + graph_dict['y-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_xlabel('\n' + graph_dict['x-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_zlabel('\n' + graph_dict['z-axis'], linespacing=3,
                        fontsize=font_size)
        colour_map = mpl.pyplot.get_cmap(graph_dict['colour map'])

        ### Other settings
        fitness = d['fitness']
        plot.w_xaxis.set_major_locator(MaxNLocator(5))
        plot.w_zaxis.set_major_locator(MaxNLocator(5))
        plot.w_yaxis.set_major_locator(MaxNLocator(5))
        plot.set_zlim3d(fitness.cost_minVal, fitness.cost_maxVal)

        ### Data
        if not (d['cost_model'] is None):
            try:
                MU = [d['cost_model'].predict(point) for point in d["z"]]
                MU = array([item[0] for item in MU])
                zi = griddata((d['x'], d['y']), MU,
                              (d['xi'][None, :], d['yi'][:, None]), method='nearest')

                norm = mpl.pyplot.matplotlib.colors.Normalize(MU_z)
                surf = plot.plot_surface(d['X'], d['Y'], zi, rstride=1, cstride=1,
                                         linewidth=0.05, antialiased=True,
                                         cmap=colour_map)
            except TypeError,e:
                logging.error('Could not create MU plot for the GPR plot')
        
    @staticmethod
    def get_attributes(name):
    
        attribute_dictionary = {
            'All':         ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'x-colour', 'o-colour',
                            'position'],
            'DesignSpace':         ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'colour map', 'x-colour', 'o-colour', 'position'],
            'Mean':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Cost_model':  ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Cost':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Progression': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'position'],
            'Fitness':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position']
        }
        return attribute_dictionary.get(name, None)
        
    @staticmethod
    def get_default_attributes():
        # Default values for describing graph visualization
        graph_title = 'Title'
        graph_names = ['Progression', 'Fitness', 'Mean', 'DesignSpace', 'Cost_model', 'Cost']

        graph_dict1 = {'subtitle': 'Currently Best Found Solution',
                       'x-axis': 'Iteration',
                       'y-axis': 'Fitness',
                       'font size': '22',
                       'position': '231'}
        graph_dict2 = {'subtitle': 'Fitness Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '232'}
        graph_dict3 = {'subtitle': 'Regression Mean',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '233'}
        graph_dict4 = {'subtitle': 'Design Space',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'font size': '22',
                       'colour map': 'gray',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '234'}
        graph_dict5 = {'subtitle': 'Cost Model',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '235'}
        graph_dict6 = {'subtitle': 'Cost Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '22',
                       'colour map': 'gray',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '236'}
        all_graph_dicts = {'Progression': graph_dict1,
                           'Fitness': graph_dict2,
                           'Mean': graph_dict3,
                           'DesignSpace': graph_dict4,
                           'Cost_model': graph_dict5,
                           'Cost': graph_dict6,
                           }
                           
        
            
        graph_dictionary = {
            'rerendering': False,
            'graph_title': graph_title,
            'graph_names': graph_names,
            'all_graph_dicts': all_graph_dicts
        }
        
        for name in graph_names:
            graph_dictionary['all_graph_dicts'][name]['generate'] = True
                          
        return graph_dictionary
 
 ## This class containts 
class MonteCarlo_ImageViewer(ImageViewer):

    @staticmethod
    def render(input_dictionary):
        logging.info("Rendering...")
        if input_dictionary["generate"]:
            dictionary = MonteCarlo_ImageViewer.get_default_attributes() ##this way the default view will be used if different one was not supplied
            figure = mpl.pyplot.figure(figsize=(24,12))
            dictionary.update(input_dictionary)
            figure.subplots_adjust(wspace=0.15, hspace=0.35)
            counter_headers = []
            header = []
            figure.suptitle(dictionary['graph_title'])

            rerender = True ## pointless... remove
            designSpace = dictionary['fitness'].designSpace
            npts = 100

            ### Initialize some graph points
            dimensions = len(designSpace)
            if dimensions == 2 :
                x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
                y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
                x, y = meshgrid(x, y)
                dictionary['x'] = reshape(x, -1)
                dictionary['y'] = reshape(y, -1)
                dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],
                                                                  dictionary['y'])])

                dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                            designSpace[0]['max'] + 0.01, npts)
                dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                            designSpace[1]['max'] + 0.01, npts)
                    
                                            
                dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],
                                                            dictionary['yi'])

                if dictionary['all_graph_dicts']['DesignSpace']['generate']:
                    #if dictionary["propa_classifier"]:
                    #    MonteCarlo_ImageViewer.plot_design_space(figure, dictionary)
                    #else:
                    MonteCarlo_ImageViewer.plot_design_space2(figure, dictionary)
                if True:
                    MonteCarlo_ImageViewer.plot_MU_S2_EI(figure, dictionary)
                if False:
                    ImageViewer.plot_fitness_function(figure, dictionary)
                if False: #dictionary['all_graph_dicts']['Cost']['generate']:
                    ImageViewer.plot_cost_function(figure, dictionary)
                if False: # dictionary['all_graph_dicts']['Progression']['generate']:
                    ImageViewer.plot_fitness_progression(figure, dictionary)
                ### Save and exit
                filename = str(dictionary['images_folder']) + '/plot' + str(dictionary['counter']) + '.png'
                if rerender and os.path.isfile(filename):
                    os.remove(filename)
                try:
                    #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                    #                                             Plot_View.DPI))
                    ImageViewer.save_fig(figure, filename, ImageViewer.DPI)
                except:
                    logging.error(
                        'MonteCarlo_ImageViewer could not render a plot',exc_info=sys.exc_info())
                #figure.tight_layout()
                mpl.pyplot.close(figure)
            elif (dimensions == 3) or (dimensions == 4):
                if designSpace[0]['type'] == "discrete":
                    x = arange(designSpace[0]['min'], designSpace[0]['max'] + designSpace[0]['step'], designSpace[0]['step'])
                else:
                    x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
                    
                if designSpace[1]['type'] == "discrete":
                    y = arange(designSpace[1]['min'], designSpace[1]['max'] + designSpace[1]['step'], designSpace[1]['step'])
                else:
                    y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
                dim_grid =[None,None]
                if designSpace[2]['type'] == "discrete":
                    dim_grid[0] = arange(designSpace[2]['min'], designSpace[2]['max'] + designSpace[2]['step'], designSpace[2]['step'])
                else: # continous
                    dim_grid[0] = linspace(designSpace[2]['min'], designSpace[2]['max'], npts)
                    
                if dimensions == 4:
                    if  designSpace[3]['type'] == "discrete":
                        dim_grid[1] = arange(designSpace[3]['min'], designSpace[3]['max'] + designSpace[3]['step'], designSpace[3]['step'])
                    else:
                        dim_grid[1] = linspace(designSpace[3]['min'], designSpace[3]['max'], npts)
                dictionary["dim_grid"] = dim_grid
                x, y = meshgrid(x, y)
                dictionary['x'] = reshape(x, -1)
                dictionary['y'] = reshape(y, -1)
                dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],
                                                                  dictionary['y'])])

                ### Define grid
                dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                            designSpace[0]['max'] + 0.01, npts)
                dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                            designSpace[1]['max'] + 0.01, npts)
                
                                            
                dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],dictionary['yi'])
            
                if True:#dictionary['all_graph_dicts']['Fitness']['generate']:
                    ImageViewer.plot_fitness_function_grid(figure, dictionary)
                if True:#dictionary['all_graph_dicts']['DesignSpace']['generate']:
                    mask = MLOImageViewer.plot_design_space_grid(figure, dictionary)
                if True:
                    MLOImageViewer.plot_MU_S2_EI_grid(figure, dictionary, mask)
            else:
                logging.info("We only support visualization of 2, 3 and 4 dimensional spaces")

            #sys.exit(0) ## I let it as a reminder... do NOT uncomment this! will get the applciation to get stuck
        else: ## do not regenerate
            pass
        
    @staticmethod
    # def plot_MU_S2_EI(figure, d):
        # logging.debug("Plotting Mean, S2, EI...")
        # if not (d['regressor'] is None):  
            # MU, S2, EI, P = d['regressor'].predict(d['z'])
            # data_ei = array([item[0] for item in EI])
            # graph_dict = d['all_graph_dicts']['EI']
            # ImageViewer.render_2d(figure, d, graph_dict, "EI", data=data_ei, fitness=d['fitness'], maxVal=data_ei.max(),minVal=data_ei.min())
            # labels = d['classifier'].predict(d['z'])
            # place(labels,labels != 1.,0.0) ### need to zero invalid and 1 valid
            # if d["propa_classifier"]:
                # data = array([item[0] for item in EI]) * (reshape(d['classifier'].predict(d['z']),-1))
                # graph_dict = d['all_graph_dicts']['EI']
                # ImageViewer.render_2d(figure, d, graph_dict, "$g(\mathbf{x})$", data=data, fitness=d['fitness'], maxVal=data.max(),minVal=data.min(),z_label="$E[I(\mathbf{x})]$")
            # else:
                # data = array([item[0] for item in EI]) * labels
                # graph_dict = d['all_graph_dicts']['EI']
                # ImageViewer.render_2d(figure, d, graph_dict, "Expected Improvement", data=data, fitness=d['fitness'], maxVal=data_ei.max(),minVal=data_ei.min(),z_label="$E[I(\mathbf{x})]$")
            # data = array([item[0] for item in MU]) * labels
            # graph_dict = d['all_graph_dicts']['Mean']
            # ImageViewer.render_2d(figure, d, graph_dict, "Predicted Throughput", data=data, fitness=d['fitness'])
            # data = array([item[0] for item in S2]) * labels
            # graph_dict = d['all_graph_dicts']['S2']
            # ImageViewer.render_2d(figure, d, graph_dict, "Uncertainty", data=data, fitness=d['fitness'], maxVal=data.max(),minVal=data.min(), z_label="$\sigma(\mathbf{x})$")
        
    #### DESIGN SPACE PLOTS
    @staticmethod
    def plot_design_space(figure, d):
        logging.info("Plotting Design Space...")
        graph_dict = d['all_graph_dicts']['DesignSpace']
        plot, save_fig  = MonteCarlo_ImageViewer.figure_wrapper(figure, graph_dict, d, "DesignSpace", three_d=False)
        fitness = d['fitness']
        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title("DESIGN SPACE",
                       fontsize=MonteCarlo_ImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(fitness.get_x_axis_name(), fontsize=font_size+12)
        plot.set_ylabel(fitness.get_y_axis_name(), fontsize=font_size+12)
        colour_map = mpl.cm.get_cmap(graph_dict['colour map'])
        xcolour = graph_dict['x-colour']
        ocolour = graph_dict['o-colour']

        ### Other settings
        '''
        locator = LinearLocator(4)
        locator.tick_values(fitness.designSpace[0]["min"], fitness.designSpace[0]["max"])
        plot.xaxis.set_major_locator(locator)
        
        locator = MaxNLocator(5,integer=True)
        locator.tick_values(fitness.designSpace[1]["min"], fitness.designSpace[1]["max"])
        plot.yaxis.set_major_locator(locator)
        
        plot.tick_params(axis='both', which='major', labelsize=20)
        '''
        ### Data
        fitness = d['fitness']
        if not (d['classifier'] is None):
            zClass = d['classifier'].predict(d['z'])
            zClass = reshape(zClass,-1)
                
            zi3 = griddata((d['x'], d['y']), zClass, (d['xi'][None, :], d['yi'][:, None]), method='nearest')
           
            CS = plot.contourf(d['X'], d['Y'],zi3,cmap=mpl.cm.gray_r, vmax=1.0, vmin=0.0)
            #CS = plot.contour(d['X'], d['Y'],zi3,linewidths=0.5,colors='k')
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(mpl.pyplot.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = mpl.pyplot.colorbar(CS, cax=cax, ticks=CS.levels)
            cbar.ax.set_yticklabels([ "                  " + v #+ " " * (int(len(v)/3))
                                     for k, v in error_labels.items()],
                                    rotation='vertical',
                                    fontsize=font_size)
            
            #
            plot_trainingset_x = [] 
            plot_trainingset_y = []
            training_set = d['classifier'].training_set
            training_labels = d['classifier'].training_labels
            
            for i in range(0, len(training_set)):
                p = training_set[i]
                plot_trainingset_x.append(p[0])
                plot_trainingset_y.append(p[1])
       
            if len(plot_trainingset_x) > 0:
                plot.scatter(x=plot_trainingset_x, y=plot_trainingset_y, c="black", marker='x', s=150, label = "Evaluations")
            plot.axis([d['X'].min(), d['X'].max(), d['Y'].min(), d['Y'].max()])
            '''
            try:
                logging.info("Transforming Axis Labels")
                x_func, y_func = MLOImageViewer.my_formatter(fitness)
                plot.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(x_func))
                plot.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(y_func))
                def two_digits(x, pos):
                    return '%1.2f' % x
            except Exception,e:
                logging.info("KRUWAAA " + str(e))
                pass
             
            plot.legend( loc='upper left', numpoints = 1)
            '''
            save_fig()
    
    #### DESIGN SPACE PLOTS
    @staticmethod
    def plot_design_space2(figure, d):
        logging.info("Plotting Design Space...")
        graph_dict = d['all_graph_dicts']['DesignSpace']
        plot, save_fig  = MonteCarlo_ImageViewer.figure_wrapper(figure, graph_dict, d, "DesignSpace", three_d=False)
        fitness = d['fitness']
        ### User settings
        font_size = int(graph_dict['font size'])+14
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MonteCarlo_ImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(fitness.get_x_axis_name(), fontsize=font_size)
        plot.set_ylabel(fitness.get_y_axis_name(), fontsize=font_size)
        colour_map = mpl.cm.get_cmap('gist_yarg')
        xcolour = graph_dict['x-colour']
        ocolour = graph_dict['o-colour']

        ### Other settings
        locator = LinearLocator(4)
        locator.tick_values(fitness.designSpace[0]["min"], fitness.designSpace[0]["max"])
        plot.xaxis.set_major_locator(locator)
        
        locator = MaxNLocator(5,integer=True)
        locator.tick_values(fitness.designSpace[1]["min"], fitness.designSpace[1]["max"])
        plot.yaxis.set_major_locator(locator)
        
        plot.tick_params(axis='both', which='major', labelsize=20)
        ### Data
        fitness = d['fitness']
        if not (d['classifier'] is None):
            zClass = (d['classifier'].predict(d['z'])-1.0)/2.0
            zi3 = griddata((d['x'], d['y']), zClass, (d['xi'][None, :], d['yi'][:, None]), method='nearest')

            #error_labels = fitness.error_labels
            error_labels = {0.0:'Valid',1.0:'Inaccuracy'}
            
            levels = [k for k, v in error_labels.items()]
            levels = [l-0.1 for l in levels]
            levels.append(levels[-1]+1.5)
            
            CS = plot.contourf(d['X'], d['Y'], zi3, levels, cmap=colour_map,  alpha = 0.5)
            
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(mpl.pyplot.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            cbar = mpl.pyplot.colorbar(CS, cax=cax, ticks=CS.levels)
            cbar.ax.set_yticklabels([ "                  " + v #+ " " * (int(len(v)/3))
                                     for k, v in error_labels.items()],
                                    rotation='vertical',
                                    fontsize=font_size)
            
            plot_trainingset_x = [] 
            plot_trainingset_y = []
            training_set = d['classifier'].training_set
            training_labels = d['classifier'].training_labels
            
            for i in range(0, len(training_set)):
                p = training_set[i]
                plot_trainingset_x.append(p[0])
                plot_trainingset_y.append(p[1])
            
            if len(plot_trainingset_x) > 0:
                plot.scatter(x=plot_trainingset_x, y=plot_trainingset_y, c="black", marker='x', s=150, label = "Evaluations")
            plot.axis([d['X'].min(), d['X'].max(), d['Y'].min(), d['Y'].max()])
            '''
            try:
                logging.info("Transforming Axis Labels")
                x_func, y_func = MLOImageViewer.my_formatter(fitness)
                plot.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(x_func))
                plot.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(y_func))
                def two_digits(x, pos):
                    return '%1.2f' % x
            except Exception,e:
                logging.info("KRUWAAA " + str(e))
                pass
            
            plot.legend( loc='upper left', numpoints = 1)
            '''
            save_fig()
    
    @staticmethod
    def get_attributes(name):
    
        attribute_dictionary = {
            'All':         ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'x-colour', 'o-colour',
                            'position'],
            'DesignSpace': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'colour map', 'x-colour', 'o-colour', 'position'],
            'Mean':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],            
            'EI':          ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],            
            'S2':          ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Cost_model':  ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Cost':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Progression': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'position'],
            'Fitness':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position']
        }
        return attribute_dictionary.get(name, None)
        
    @staticmethod
    def get_default_attributes():
        # Default values for describing graph visualization
        graph_title = 'Title'
        graph_names = ['Progression', 'Fitness', 'Mean', 'EI', 'S2', 'DesignSpace', 'Cost_model', 'Cost']

        graph_dict1 = {'subtitle': 'Currently Best Found Solution',
                       'x-axis': 'Iteration',
                       'y-axis': 'FITNESS',
                       'font size': '22',
                       'position': '241'}
        graph_dict2 = {'subtitle': 'FITNESS FUNCTION',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'FITNESS',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '242'}
        graph_dict3 = {'subtitle': 'REGRESSION MEAN',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '243'}
        graph_dict4 = {'subtitle': 'Regression EI',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '244'}
        graph_dict5 = {'subtitle': 'Regression S2',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '245'}
        graph_dict6 = {'subtitle': 'DESIGN SPACE',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'font size': '22',
                       'colour map': 'gray',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '246'}
        graph_dict7 = {'subtitle': 'Cost Model',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '247'}
        graph_dict8 = {'subtitle': 'Cost Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '22',
                       'colour map': 'gray',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '248'}
        all_graph_dicts = {'Progression': graph_dict1,
                           'Fitness': graph_dict2,
                           'Mean': graph_dict3,
                           'EI': graph_dict4,
                           'S2': graph_dict5,
                           'DesignSpace': graph_dict6,
                           'Cost_model': graph_dict7,
                           'Cost': graph_dict8,
                           }
                           
        
            
        graph_dictionary = {
            'rerendering': False,
            'graph_title': graph_title,
            'graph_names': graph_names,
            'all_graph_dicts': all_graph_dicts
        }
        
        for name in graph_names:
            graph_dictionary['all_graph_dicts'][name]['generate'] = True
                          
        return graph_dictionary
        
        
class P_ARDEGO_Trial_ImageViewer(MonteCarlo_ImageViewer):
    
    @staticmethod
    def get_default_attributes():
        # Default values for describing graph visualization
        graph_title = 'Title'
        graph_names = ['Progression', 'Fitness', 'Mean', 'EI', 'S2', 'DesignSpace', 'Cost_model', 'Cost']

        graph_dict1 = {'subtitle': 'Currently Best Found Solution',
                       'x-axis': 'Iteration',
                       'y-axis': 'Fitness',
                       'font size': '22',
                       'position': '241'}
        graph_dict2 = {'subtitle': 'Fitness Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '20',
                       'colour map': 'gray',
                       'position': '242'}
        graph_dict3 = {'subtitle': 'Regression Mean',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '243'}
        graph_dict4 = {'subtitle': 'Regression EI',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '244'}
        graph_dict5 = {'subtitle': 'Regression S2',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '245'}
        graph_dict6 = {'subtitle': 'Design Space',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'font size': '22',
                       'colour map': 'gray',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '246'}
        graph_dict7 = {'subtitle': 'Cost Model',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '22',
                       'colour map': 'gray',
                       'position': '247'}
        graph_dict8 = {'subtitle': 'Cost Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '22',
                       'colour map': 'gray',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '248'}
        all_graph_dicts = {'Progression': graph_dict1,
                           'Fitness': graph_dict2,
                           'Mean': graph_dict3,
                           'EI': graph_dict4,
                           'S2': graph_dict5,
                           'DesignSpace': graph_dict6,
                           'Cost_model': graph_dict7,
                           'Cost': graph_dict8,
                           }
                           
        
            
        graph_dictionary = {
            'rerendering': False,
            'graph_title': graph_title,
            'graph_names': graph_names,
            'all_graph_dicts': all_graph_dicts
        }
        
        for name in graph_names:
            graph_dictionary['all_graph_dicts'][name]['generate'] = True
                          
        return graph_dictionary
        
 

 