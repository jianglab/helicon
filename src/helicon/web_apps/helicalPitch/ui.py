from shiny import reactive, req
from shiny.express import input, ui, render, expressify
from shinywidgets import render_plotly

import helicon
import compute

@expressify
def display_app_ui(images=reactive.value([]), image_labels=reactive.value([]), displayed_class_ids=reactive.value([]), selected_helix_lengths=reactive.value([]), pair_distances=reactive.value([])):
  
  image_size = reactive.value(None)
  @reactive.effect
  @reactive.event(images)
  def _():
      req(len(images())>0)
      image_size.set(max(images()[0].shape))
  
  ui.head_content(ui.tags.title("HelicalPitch"))

  with ui.sidebar(width=421): # 456
    ui.input_radio_buttons("input_mode_params", "How to obtain the Class2D parameter file:", 
                          choices=["upload", "url"], 
                          selected="url",
                          inline=True
                          )
    with ui.panel_conditional("input.input_mode_params === 'upload'"):
      ui.input_file("upload_params", "Upload the class2d parameters in a RELION star or cryoSPARC cs file", 
                  accept=['.star', '.cs'])
    
    with ui.panel_conditional("input.input_mode_params === 'url'"):
      ui.input_text("url_params", "Download URL for a RELION star or cryoSPARC cs file", 
                  value="https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/768px/run_it020_data.star")
    
    ui.input_radio_buttons("input_mode_classes", "How to obtain the class average images:", 
                          choices=["upload", "url"], 
                          selected="url",
                          inline=True
                          )
    with ui.panel_conditional("input.input_mode_classes === 'upload'"):
      ui.input_file("upload_classes", "Upload the class averages in MRC format (.mrcs, .mrc)", 
                  accept=['.mrcs', '.mrc'])
    
    with ui.panel_conditional("input.input_mode_classes === 'url'"):
      ui.input_text("url_classes", "Download URL for a RELION or cryoSPARC Class2D output mrc(s) file", 
                  value="https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/768px/run_it020_classes.mrcs")

    with ui.div(style="max-height: 50vh; overflow-y: auto;"):
      selected_classes = helicon.shiny.image_select(id="select_classes", label="Select classe(s):", images=images, image_labels=image_labels, image_size=reactive.value(128), auto_select_first=True)

      images_selected = reactive.value([])
      image_labels_selected = reactive.value([])
      @reactive.effect
      def _():
        images_selected.set([images()[i] for i in selected_classes()])
        image_labels_selected.set([image_labels()[i] for i in selected_classes()])

    with ui.layout_columns(col_widths=[6, 6]):
      ui.input_checkbox("ignore_blank", "Ignore blank classes", value=True)
      ui.input_checkbox("sort_abundance", "Sort the classes by abundance", value=True)
    
    with ui.layout_columns(col_widths=[6, 6], style="align-items: flex-end;"):
      ui.input_numeric("apix_particle", "Pixel size of particles (Å/pixel)", min=0.1, max=100.0, value=None, step=0.001)
      ui.input_numeric("apix_micrograph", "Pixel size of micrographs for particle extraction (Å/pixel)", min=0.1, max=100.0, value=None, step=0.001)

    @render.ui
    def display_warning_apix_micrograph():
          msg = "Please carefully verify the pixel size of the micrographs used for particle picking/extraction."
          return ui.markdown(f"<span style='color: red;'>{msg}</span>")

  title = "HelicalPitch: determine helical pitch/twist using 2D Classification info"
  ui.h1(title)
  
  with ui.layout_columns(col_widths=(5, 7, 12)):
    with ui.card():
      helicon.shiny.image_select(id="selected_classes", label="Selected classe(s):", images=images_selected, image_labels=image_labels_selected, image_size=image_size, disable_selection=True)
      
      with ui.layout_columns(col_widths=[12, 12], style="align-items: flex-end;"):
        with ui.card():
          @render_plotly
          @reactive.event(selected_helix_lengths)
          def lengths_histogram_display():
            class_indices = [str(displayed_class_ids()[i]+1) for i in selected_classes()]
            log_y = True
            title = f"Filament Lengths: Class {' '.join(class_indices)}"
            xlabel = "Filament Legnth (Å)"
            ylabel = "# of Filaments"
            nbins = 50 #input.bins()
            fig = compute.plot_histogram(data=selected_helix_lengths(), title=title, xlabel=xlabel, ylabel=ylabel, bins=nbins, log_y=log_y)
        
            return fig

        with ui.card():
          with ui.layout_columns(col_widths=6, style="align-items: flex-end;"):
            ui.input_numeric("min_len", "Minimal length (Å)", min=0.0, value=0, step=1.0)
            ui.input_numeric("max_len", "Maximal length (Å)", min=-1, value=-1, step=1.0)
            ui.input_numeric("bins", "Number of histogram bins", min=1, value=100, step=1)
            ui.input_numeric("max_pair_dist", "Maximal pair distance (Å) to plot", min=-1, value=-1, step=1.0)
            ui.input_numeric("rise", "Helical rise (Å)", min=0.01, max=100.0, value=4.75, step=0.01)
    
    with ui.card():
      @render_plotly
      @reactive.event(pair_distances)
      def pair_distances_histogram_display():
        class_indices = [str(displayed_class_ids()[i]+1) for i in selected_classes()]
        rise = 4.75 #input.rise()
        log_y = True
        title = f"Pair Distances: Class {' '.join(class_indices)}"
        xlabel = "Pair Distance (Å)"
        ylabel = "# of Pairs"
        nbins = 100 #input.bins()
        
        fig = compute.plot_histogram(data=pair_distances(), title=title, xlabel=xlabel, ylabel=ylabel, max_pair_dist=None, bins=nbins, log_y=log_y, show_pitch_twist=dict(rise=rise, csyms=(1,2,3,4)), multi_crosshair=True)

        return fig
      
      ui.markdown("**How to interpretate the histogram:** an informative histogram should have clear peaks with equal spacing. If so, hover your mouse pointer on the first prominent peak to the first major peak off the origin to align the vertial lines well with the peaks. Once you have decided on that line postion, read the hover text which shows the twist values assuming the pair-distance is the helical pitch (adjusted for the cyclic symmetries around the helical axis). If the histogram does not show clear peaks, it indicates that the Class2D quality is bad. You might consider changing the 'Minimal length (Å)' from 0 to a larger value (for example, 1000 Å) to improve the peaks in the histogram. If that does not help, you might consider redoing the Class2D task with longer extracted segments (>0.5x helical pitch) from longer filaments (> 1x pitch)")

    ui.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu/HelicalPitch). Report problems to [HelicalPitch@GitHub](https://github.com/jianglab/HelicalPitch/issues)*")
  
  return selected_classes