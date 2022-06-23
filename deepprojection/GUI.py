import os
import tkinter as tk
import webbrowser
from tkinter import ttk
from tkinter.messagebox import askyesno

import numpy as np
import tkfilebrowser

from deepprojection import MaxProjection, Project, get_stack_directories


class Projector(tk.Tk):
    """Graphical user interface for DeepProjection"""

    def __init__(self):
        tk.Tk.__init__(self)
        self.geometry('600x430')
        self.resizable(0, 0)
        self.protocol("WM_DELETE_WINDOW", self.confirm_quit)

        # add icon here (thumbnail)

        # variables
        self.joblist = []
        self.save_dir = tk.StringVar(value='-results saved in parent directory-')
        self.weights = tk.StringVar(value='-select DeepProjection weights-')
        self.data_format = tk.StringVar(value='TZXY')
        self.mode = tk.StringVar(value='mip')
        self.mask_thrs = tk.StringVar(value='None')
        self.filter_size = tk.StringVar(value='0')
        self.time_average = tk.StringVar(value='0')
        self.offset = tk.StringVar(value='0')
        self.channel = tk.StringVar(value='None')
        self.resize_dim = tk.StringVar(value='1024, 1024')
        self.clip_thrs = tk.StringVar(value='0, 99.98')
        self.normalization_mode = tk.StringVar(value='movie')
        self.temp_dir = tk.StringVar(value='./temp/')
        self.invert_slices = tk.IntVar(value=False)
        self.bigtiff = tk.IntVar(value=False)

        # build
        self.build_statusbar()
        self.build_path_handler()
        self.build_detector()

    # gui
    def build_path_handler(self):
        self.lf_selector = tk.LabelFrame(self, text='Selection of stacks and directories')
        self.lf_selector.place(x=5, y=5, width=590, height=185)
        # joblist
        # stack browser (choose single stacks / .tif files)
        tk.Button(self.lf_selector, text='Browse stacks', command=self.select_stacks).place(x=10, y=5, width=120)
        # directory browser (choose single or multiple directories)
        tk.Button(self.lf_selector, text='Browse directories', command=self.select_dirs).place(x=10, y=35, width=120)
        # get all subdirectories with tif files
        tk.Button(self.lf_selector, text='Find all directories', command=self.get_subdirectories).place(x=10, y=65,
                                                                                                        width=120)
        # remove selected stacks / dirs
        tk.Button(self.lf_selector, text='Remove item', command=self.delete_job).place(x=10, y=95, width=120)
        # clear list
        tk.Button(self.lf_selector, text='Clear list', command=self.clear_joblist).place(x=10, y=125, width=120)
        # list of paths (ability to add and remove)
        self.listbox = tk.Listbox(self.lf_selector)
        self.listbox.place(x=145, y=5, height=150, width=410)
        self.scrollbar = tk.Scrollbar(self.lf_selector)
        self.scrollbar.place(x=556, y=5, height=150)
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)

        # directory browser (choose single or multiple directories)

    # statusbar
    def build_statusbar(self):
        ttk.Separator(self, orient='horizontal').place(x=0, y=409, relwidth=1)
        self.status = tk.StringVar(value='Ready')
        self.statusbar = tk.Label(self, textvariable=self.status, anchor=tk.W, fg='white', background='green')
        self.statusbar.place(x=0, y=410)
        link = tk.Label(self, text="Project homepage", fg="blue", cursor="hand2")
        link.place(x=490, y=410)
        link.bind("<Button-1>", lambda e: webbrowser.open_new('https://github.com/danihae/DeepProjection/'))

    def build_converter(self):
        pass

    def build_detector(self):
        lf_detector = tk.LabelFrame(self, text='DeepProjection prediction')
        lf_detector.place(x=5, y=190, width=590, height=215)
        # save dir
        tk.Label(lf_detector, text='Save directory:').place(x=5, y=10)
        self.ent_save_dir = tk.Entry(lf_detector, textvariable=self.save_dir)
        self.ent_save_dir.place(x=120, y=10, width=400)
        self.button_save_dir = tk.Button(lf_detector, text='Browse', command=self.select_save_dir)
        self.button_save_dir.place(x=530, y=8)
        # browse network weights
        tk.Label(lf_detector, text='Network weights:').place(x=5, y=40)
        self.ent_network_weights = tk.Entry(lf_detector, textvariable=self.weights)
        self.ent_network_weights.place(x=120, y=40, width=400)
        tk.Button(lf_detector, text='Browse', command=self.select_weights).place(x=530, y=38)
        # parameters
        # data format
        tk.Label(lf_detector, text='Data format:').place(x=5, y=70)
        self.combo_data_format = ttk.Combobox(lf_detector, textvariable=self.data_format,
                                              values=['ZXY', 'Z-XY', 'TZXY', 'T-ZXY', 'T-Z-XY', 'ZTCXY', 'ZCXY'])
        self.combo_data_format.place(x=80, y=70, width=60)
        # resize dimensions
        tk.Label(lf_detector, text='Resize dims. [px]:').place(x=143, y=70)
        self.ent_resize_dim = tk.Entry(lf_detector, textvariable=self.resize_dim)
        self.ent_resize_dim.place(x=240, y=70, width=60)
        # clip threshold
        tk.Label(lf_detector, text='Clip thres. [%]:').place(x=305, y=70)
        self.ent_clip_thrs = tk.Entry(lf_detector, textvariable=self.clip_thrs)
        self.ent_clip_thrs.place(x=390, y=70, width=45)
        # normalization mode
        tk.Label(lf_detector, text='Norm. mode:').place(x=440, y=70)
        self.combo_normalization_mode = ttk.Combobox(lf_detector, textvariable=self.normalization_mode,
                                                     values=['movie', 'stack', 'first'])
        self.combo_normalization_mode.place(x=520, y=70, width=60)
        self.combo_normalization_mode.set('movie')
        # mode
        tk.Label(lf_detector, text='Proj. mode:').place(x=5, y=100)
        self.combo_mode = ttk.Combobox(lf_detector, textvariable=self.mode,
                                       values=['mip', 'max', 'min', 'mean', 'median'])
        self.combo_mode.place(x=80, y=100, width=50)
        self.combo_mode.set('mip')
        # mask threshold
        tk.Label(lf_detector, text='Mask thres.:').place(x=143, y=100)
        self.ent_mask_thrs = tk.Entry(lf_detector, textvariable=self.mask_thrs)
        self.ent_mask_thrs.place(x=220, y=100, width=40)
        # time average
        tk.Label(lf_detector, text='Time avg.:').place(x=265, y=100)
        self.combo_time_average = ttk.Combobox(lf_detector, textvariable=self.time_average,
                                               values=['0', '3', '5', '9'])
        self.combo_time_average.place(x=330, y=100, width=35)
        self.combo_time_average.set('0')
        # offset
        tk.Label(lf_detector, text='Offset:').place(x=375, y=100)
        self.ent_offset = tk.Entry(lf_detector, textvariable=self.offset)
        self.ent_offset.place(x=420, y=100, width=45)
        # color channel
        tk.Label(lf_detector, text='Channel:').place(x=470, y=100)
        self.combo_channel = ttk.Combobox(lf_detector, textvariable=self.channel,
                                          values=['None', '0', '1', '2'])
        self.combo_channel.place(x=530, y=100, width=50)
        self.combo_channel.set('None')
        # temp folder
        tk.Label(lf_detector, text='Temp. directory:').place(x=5, y=130)
        self.ent_temp_dir = tk.Entry(lf_detector, textvariable=self.temp_dir)
        self.ent_temp_dir.place(x=120, y=132, width=220)
        self.button_temp_dir = tk.Button(lf_detector, text='Browse', command=self.select_temp_dir)
        self.button_temp_dir.place(x=350, y=128)
        # invert slices
        self.radio_invert_slices = tk.Checkbutton(lf_detector, text='Invert slices', variable=self.invert_slices,
                                                  onvalue=1, offvalue=0)
        self.radio_invert_slices.place(x=420, y=130)
        # bigtiff
        self.radio_bigtiff = tk.Checkbutton(lf_detector, text='BigTIFF', variable=self.bigtiff,
                                            onvalue=1, offvalue=0)
        self.radio_bigtiff.place(x=510, y=130)
        # buttons for prediction
        self.button_detect = tk.Button(lf_detector, text='Predict with DeepProjection', command=self.predict)
        self.button_detect.place(x=5, y=165, width=280)
        self.button_max_int = tk.Button(lf_detector, text='Maximum intensity projection', command=self.max_projection)
        self.button_max_int.place(x=300, y=165, width=280)

    # set status bar idle or ready
    def set_idle(self):
        self.status.set('Idle...')
        self.statusbar.config(bg='red')
        self.update()

    def set_ready(self):
        self.status.set('Ready')
        self.statusbar.config(bg='green')
        self.update()

    def confirm_quit(self):
        answer = askyesno(title='Quit DeepProjector',
                          message='Are you sure that you want to quit?')
        if answer:
            self.destroy()

    def select_stacks(self):
        add_stacks = tkfilebrowser.askopenfilenames(title='Select stacks', okbuttontext='Add',
                                                    initialdir='../')
        add_stacks = [path.replace('\\', '/') for path in add_stacks]
        self.joblist.extend(add_stacks)
        self.update_joblist()

    def select_dirs(self):
        add_paths = tkfilebrowser.askopendirnames(title='Select directories', okbuttontext='Add',
                                                  initialdir='../')
        add_paths = [path.replace('\\', '/') + '/' for path in add_paths]
        self.joblist.extend(add_paths)
        self.update_joblist()

    def update_joblist(self):
        self.listbox.delete(0, tk.END)
        for job in self.joblist:
            self.listbox.insert(tk.END, job)

    def get_subdirectories(self):
        base_folders = tkfilebrowser.askopendirnames(title='Select base-directories', okbuttontext='Find stack folders')
        for folder in base_folders:
            add_paths = get_stack_directories(folder)
            add_paths = [path.replace('\\', '/') for path in add_paths]
            add_paths = np.unique(add_paths)
            self.joblist.extend(add_paths)
        self.update_joblist()

    def delete_job(self):
        idx_selected = self.listbox.curselection()[0]
        self.joblist.pop(idx_selected)
        self.update_joblist()

    def clear_joblist(self):
        self.joblist = []
        self.update_joblist()

    def select_save_dir(self):
        save_dir = tkfilebrowser.askopendirname(title='-results saved in parent directory-',
                                                okbuttontext='Select', initialdir='../')
        save_dir = save_dir.replace('\\', '/') + '/'
        self.save_dir.set(save_dir)

    def select_temp_dir(self):
        temp_dir = tkfilebrowser.askopendirname(title='Select temp. directory', okbuttontext='Select',
                                                initialdir='../')
        temp_dir = temp_dir.replace('\\', '/') + '/'
        self.temp_dir.set(temp_dir)

    def select_weights(self):
        path_weights = tkfilebrowser.askopenfilename(title='Select DeepProjection weights', okbuttontext='Select',
                                                     initialdir='../trained_networks/')
        path_weights = path_weights.replace('\\', '/')
        self.weights.set(path_weights)

    def predict(self):
        self.set_idle()

        for i, job in enumerate(self.joblist):
            if str(self.save_dir.get()) == '-results saved in parent directory-':
                filename_output = None
            else:
                filename_output = str(self.save_dir.get()) + os.path.basename(job[:-1]) + '.tif'
            try:
                self.listbox.itemconfig(i, bg='blue', fg='white')
                self.update()
                Project(job, weights=str(self.weights.get()),
                        data_format=str(self.data_format.get()),
                        mode=str(self.mode.get()),
                        mask_thrs=eval(self.mask_thrs.get()),
                        filter_time=eval(self.time_average.get()),
                        offset=eval(self.offset.get()),
                        channel=eval(self.channel.get()),
                        filename_output=filename_output,
                        resize_dim=eval(self.resize_dim.get()),
                        clip_thrs=eval(self.clip_thrs.get()),
                        normalization_mode=str(self.combo_normalization_mode.get()),
                        temp_folder=str(self.temp_dir.get()),
                        bigtiff=bool(self.bigtiff.get()), invert_slices=bool(self.invert_slices.get()))
                self.listbox.itemconfig(i, bg='green', fg='white')
                self.update()
            except Exception as e:
                print(f'ERROR: {dir} could not be predicted.')
                print(e)
                self.listbox.itemconfig(i, bg='red', fg='white')
                self.update()
        self.set_ready()

    def max_projection(self):
        self.set_idle()
        for i, job in enumerate(self.joblist):
            if str(self.save_dir.get()) == '-predicted movies saved in parent directory-':
                filename_output = job[:-1] + '_MAX.tif'
            else:
                filename_output = str(self.save_dir.get()) + os.path.basename(dir[:-1]) + '.tif'
            try:
                self.listbox.itemconfig(i, bg='blue', fg='white')
                self.update()
                MaxProjection(dir, filename_output=filename_output, bigtiff=bool(self.bigtiff.get()))
                self.listbox.itemconfig(i, bg='green', fg='white')
                self.update()
            except:
                print(f'ERROR: maximum intensity projection for {dir} failed.')
                self.listbox.itemconfig(i, bg='red', fg='white')
                self.update()
        self.set_ready()


if __name__ == "__main__":
    app = Projector()
    app.title("DeepProjector")
    app.mainloop()
