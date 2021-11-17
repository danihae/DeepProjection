import os
import tkinter as tk
import webbrowser
from tkinter import ttk
from tkinter.messagebox import askyesno

import numpy as np
import tkfilebrowser

from deepprojection import MaxProjection, PredictMovie, PredictStack, ProjNet, get_stack_directories


class Projector(tk.Tk):
    """Graphical user interface for DeepProjection"""
    def __init__(self):
        tk.Tk.__init__(self)
        self.geometry('600x400')
        self.resizable(0, 0)
        self.protocol("WM_DELETE_WINDOW", self.confirm_quit)

        # add icon here (thumbnail)

        # variables
        self.dirnames, self.stacknames = [], []
        self.save_dir = tk.StringVar(value='-select save directory-')
        self.weights = tk.StringVar(value='-select DeepProjection weights-')
        self.resize_dim = tk.StringVar(value='1024, 1024')
        self.clip_thrs = tk.StringVar(value='99.98')
        self.normalization_mode = tk.StringVar(value='movie')

        # build
        self.build_statusbar()
        self.build_path_handler()
        self.build_detector()

        print()

    # gui
    def build_path_handler(self):
        # notebook with two tabs (stacks and movies)
        self.nb_paths = ttk.Notebook(self)
        self.nb_paths.place(x=5, y=5, width=590, height=180)
        self.nb_paths.bind('<<NotebookTabChanged>>', self.switch_tab_stacks_movies)
        # single stacks
        self.tab_stacks = ttk.Frame(self.nb_paths)
        self.nb_paths.add(self.tab_stacks, text='Single stacks')
        # stack browser (choose single or stacks)
        tk.Button(self.tab_stacks, text='Browse stacks', command=self.select_stacks).place(x=10, y=20, width=120)
        # remove selected stack
        tk.Button(self.tab_stacks, text='Remove stack', command=self.delete_stack).place(x=10, y=60, width=120)
        # clear list
        tk.Button(self.tab_stacks, text='Clear list', command=self.clear_stacks).place(x=10, y=100, width=120)
        # list of paths (ability to add and remove)
        self.listbox_stacks = tk.Listbox(self.tab_stacks)
        self.listbox_stacks.place(x=145, y=5, height=140, width=410)
        self.scrollbar_stacks = tk.Scrollbar(self.tab_stacks)
        self.scrollbar_stacks.place(x=556, y=5, height=140)
        self.listbox_stacks.config(yscrollcommand=self.scrollbar_stacks.set)
        self.scrollbar_stacks.config(command=self.listbox_stacks.yview)

        # movies
        self.tab_movies = ttk.Frame(self.nb_paths)
        self.nb_paths.add(self.tab_movies, text='Movies')
        # directory browser (choose single or multiple directories)
        tk.Button(self.tab_movies, text='Browse directories', command=self.select_dirs).place(x=10, y=20, width=120)
        # get all subdirectories with tif files
        tk.Button(self.tab_movies, text='Find all directories', command=self.get_subdirectories).place(x=10, y=50,
                                                                                                       width=120)
        # remove selected dir path
        tk.Button(self.tab_movies, text='Remove directory', command=self.delete_dir).place(x=10, y=80, width=120)
        # clear list
        tk.Button(self.tab_movies, text='Clear list', command=self.clear_dirs).place(x=10, y=110, width=120)
        # list of paths (ability to add and remove)
        self.listbox_movies = tk.Listbox(self.tab_movies)
        self.listbox_movies.place(x=145, y=5, height=140, width=410)
        self.scrollbar_movies = tk.Scrollbar(self.tab_movies)
        self.scrollbar_movies.place(x=556, y=5, height=140)
        self.listbox_movies.config(yscrollcommand=self.scrollbar_movies.set)
        self.scrollbar_movies.config(command=self.listbox_movies.yview)

    # statusbar
    def build_statusbar(self):
        ttk.Separator(self, orient='horizontal').place(x=0, y=379, relwidth=1)
        self.status = tk.StringVar(value='Ready')
        self.statusbar = tk.Label(self, textvariable=self.status, anchor=tk.W, fg='white', background='green')
        self.statusbar.place(x=0, y=380)
        link = tk.Label(self, text="Project homepage", fg="blue", cursor="hand2")
        link.place(x=490, y=380)
        link.bind("<Button-1>", lambda e: webbrowser.open_new('https://gitlab.oit.duke.edu/dh287/projnet'))

    def build_converter(self):
        pass

    def build_detector(self):
        lf_detector = tk.LabelFrame(self, text='DeepProjection prediction')
        lf_detector.place(x=5, y=190, width=590, height=180)
        # save dir
        tk.Label(lf_detector, text='Save directory:').place(x=5, y=10)
        self.ent_save_dir = tk.Entry(lf_detector, textvar=self.save_dir)
        self.ent_save_dir.place(x=120, y=10, width=400)
        self.button_save_dir = tk.Button(lf_detector, text='Browse', command=self.select_save_dir)
        self.button_save_dir.place(x=530, y=8)
        # browse network weights
        tk.Label(lf_detector, text='Network weights:').place(x=5, y=40)
        self.ent_network_weights = tk.Entry(lf_detector, textvar=self.weights)
        self.ent_network_weights.place(x=120, y=40, width=400)
        tk.Button(lf_detector, text='Browse', command=self.select_weights).place(x=530, y=38)
        # parameter
        # resize dimensions
        tk.Label(lf_detector, text='Resize dimensions [px]:').place(x=5, y=70)
        self.ent_resize_dim = tk.Entry(lf_detector, textvar=self.resize_dim)
        self.ent_resize_dim.place(x=140, y=70, width=80)
        # clip threshold
        tk.Label(lf_detector, text='Clip threshold [%]:').place(x=230, y=70)
        self.ent_clip_thrs = tk.Entry(lf_detector, textvar=self.clip_thrs)
        self.ent_clip_thrs.place(x=340, y=70, width=40)
        # normalization mode
        tk.Label(lf_detector, text='Normalization mode:').place(x=385, y=70)
        self.combo_normalization_mode = ttk.Combobox(lf_detector, textvariable=self.normalization_mode,
                                                     values=['movie', 'stack'])
        self.combo_normalization_mode.place(x=510, y=70, width=60)
        self.combo_normalization_mode.set('movie')
        # buttons for prediction
        self.button_detect = tk.Button(lf_detector, text='Predict with DeepProjection', command=self.predict)
        self.button_detect.place(x=5, y=120, width=280)
        self.button_max_int = tk.Button(lf_detector, text='Maximum intensity projection', command=self.max_projection)
        self.button_max_int.place(x=300, y=120, width=280)

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
        self.stacknames.extend(add_stacks)
        self.update_stacklist()

    def select_dirs(self):
        add_paths = tkfilebrowser.askopendirnames(title='Select directories', okbuttontext='Add',
                                                  initialdir='../')
        add_paths = [path.replace('\\', '/') + '/' for path in add_paths]
        self.dirnames.extend(add_paths)
        self.update_dirlist()

    def update_stacklist(self):
        self.listbox_stacks.delete(0, tk.END)
        for stack in self.stacknames:
            self.listbox_stacks.insert(tk.END, stack)

    def update_dirlist(self):
        self.listbox_movies.delete(0, tk.END)
        for dir in self.dirnames:
            self.listbox_movies.insert(tk.END, dir)

    def get_subdirectories(self):
        base_folder = tkfilebrowser.askopendirname(title='Select base-directory', okbuttontext='Find stack folders')
        add_paths = get_stack_directories(base_folder, signature='.tif')
        add_paths = [path.replace('\\', '/') for path in add_paths]
        add_paths = np.unique(add_paths)
        self.dirnames.extend(add_paths)
        self.update_dirlist()

    def delete_stack(self):
        idx_selected = self.listbox_stacks.curselection()[0]
        self.stacknames.pop(idx_selected)
        self.update_stacklist()

    def delete_dir(self):
        idx_selected = self.listbox_movies.curselection()[0]
        self.dirnames.pop(idx_selected)
        self.update_dirlist()

    def clear_dirs(self):
        self.dirnames = []
        self.update_dirlist()

    def clear_stacks(self):
        self.stacknames = []
        self.update_stacklist()

    def switch_tab_stacks_movies(self, event=None):
        if self.nb_paths.index('current') == 0:
            self.combo_normalization_mode['state'] = tk.DISABLED
            self.ent_save_dir['state'] = tk.NORMAL
            self.save_dir.set('-select save directory-')
            self.button_save_dir['state'] = tk.NORMAL
            self.button_max_int['state'] = tk.DISABLED
        elif self.nb_paths.index('current') == 1:
            self.combo_normalization_mode['state'] = tk.NORMAL
            self.ent_save_dir['state'] = tk.DISABLED
            self.button_save_dir['state'] = tk.DISABLED
            self.button_max_int['state'] = tk.NORMAL
            self.save_dir.set('-predicted movies saved in parent directory-')
        self.update()

    def select_save_dir(self):
        save_dir = tkfilebrowser.askopendirname(title='Select save directory', okbuttontext='Select',
                                                initialdir='../')
        save_dir = save_dir.replace('\\', '/') + '/'
        self.save_dir.set(save_dir)

    def select_weights(self):
        path_weights = tkfilebrowser.askopenfilename(title='Select DeepProjection weights', okbuttontext='Select',
                                                     initialdir='../trained_networks/')
        path_weights = path_weights.replace('\\', '/')
        self.weights.set(path_weights)

    def predict(self):
        self.set_idle()
        if self.nb_paths.index('current') == 0:
            for i, stack in enumerate(self.stacknames):
                try:
                    self.listbox_stacks.itemconfig(i, bg='blue', fg='white')
                    self.update()
                    save_name = str(self.save_dir.get()) + os.path.basename(stack)
                    print(stack)
                    PredictStack(stack, filename_output=save_name, model=ProjNet, weights=str(self.weights.get()),
                                 resize_dim=eval(self.resize_dim.get()),
                                 clip_thrs=float(self.clip_thrs.get()))
                    self.listbox_stacks.itemconfig(i, bg='green', fg='white')
                    self.update()
                except Exception as e:
                    print(f'ERROR: {stack} could not be predicted.')
                    print(e)
                    self.listbox_stacks.itemconfig(i, bg='red', fg='white')
                    self.update()
        elif self.nb_paths.index('current') == 1:
            for i, dir in enumerate(self.dirnames):
                try:
                    self.listbox_movies.itemconfig(i, bg='blue', fg='white')
                    self.update()
                    PredictMovie(dir, model=ProjNet, weights=str(self.weights.get()),
                                 resize_dim=eval(self.resize_dim.get()),
                                 clip_thrs=float(self.clip_thrs.get()),
                                 normalization_mode=str(self.combo_normalization_mode.get()))
                    self.listbox_movies.itemconfig(i, bg='green', fg='white')
                    self.update()
                except Exception as e:
                    print(f'ERROR: {dir} could not be predicted.')
                    print(e)
                    self.listbox_movies.itemconfig(i, bg='red', fg='white')
                    self.update()
        self.set_ready()

    def max_projection(self):
        self.set_idle()
        if self.nb_paths.index('current') == 0:
            for i, dir in enumerate(self.dirnames):
                try:
                    self.listbox_movies.itemconfig(i, bg='blue', fg='white')
                    self.update()
                    res_name = dir[:-1] + '_MAX.tif'
                    MaxProjection(dir, res_name)
                    self.listbox_movies.itemconfig(i, bg='green', fg='white')
                    self.update()
                except:
                    print(f'ERROR: maximum intensity projection for {dir} failed.')
                    self.listbox_movies.itemconfig(i, bg='red', fg='white')
                    self.update()
        elif self.nb_paths.index('current') == 1:
            for i, stack in enumerate(self.stacknames):
                try:
                    self.listbox_stacks.itemconfig(i, bg='blue', fg='white')
                    self.update()
                    res_name = stack[:-1] + '_MAX.tif'
                    MaxProjection(stack, res_name)
                    self.listbox_stacks.itemconfig(i, bg='green', fg='white')
                    self.update()
                except:
                    print(f'ERROR: maximum intensity projection for {stack} failed.')
                    self.listbox_stacks.itemconfig(i, bg='red', fg='white')
                    self.update()
        self.set_ready()


if __name__ == "__main__":
    app = Projector()
    app.title("DeepProjector")
    app.mainloop()
