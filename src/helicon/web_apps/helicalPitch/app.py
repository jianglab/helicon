import numpy as np
import ui as ui_app
import compute

from shiny import reactive, req
from shiny.express import input, ui

params_orig = reactive.value(None)
params_work = reactive.value(None)
apix_micrograph_auto = reactive.value(0)
apix_micrograph_auto_source = reactive.value(None)
apix_micrograph = reactive.value(0)
apix_particle = reactive.value(0)

data_all = reactive.value(None)
apix_class = reactive.value(0)
image_size = reactive.value(0)

displayed_class_ids = reactive.value([])
displayed_class_images = reactive.value([])
displayed_class_labels = reactive.value([])

selected_helices = reactive.value([])
selected_helix_lengths = reactive.value([])
retained_helices_by_length = reactive.value([])
pair_distances = reactive.value([])

selected_classes = ui_app.display_app_ui(images=displayed_class_images, image_labels=displayed_class_labels, displayed_class_ids=displayed_class_ids, selected_helix_lengths=selected_helix_lengths, pair_distances=pair_distances)

@reactive.effect
@reactive.event(input.upload_classes)
def get_class2d_from_upload():
  req(input.input_mode_classes()=="upload")
  fileinfo = input.upload_classes()
  class_file = fileinfo[0]['datapath']
  try:
    data, apix = compute.get_class2d_from_file(class_file)
    nx = data.shape[-1]
  except:
    data, apix = None, 0
    nx = 0
    m = ui.modal(
      f"failed to read the uploaded 2D class average images from {fileinfo[0]['name']}",
      title="File upload error",
      easy_close=True,
      footer=None,
    )
    ui.modal_show(m)
  data_all.set(data)
  apix_class.set(apix)
  image_size.set(nx)

@reactive.effect
@reactive.event(input.url_classes)
def get_class2d_from_url():
  req(input.input_mode_classes()=="url")
  req(len(input.url_classes())>0)
  url = input.url_classes()
  try:
    data, apix = compute.get_class2d_from_url(url)
    nx = data.shape[-1]
  except:
    data, apix = None, 0
    nx = 0
    m = ui.modal(
      f"failed to download 2D class average images from {input.url_classes()}",
      title="File download error",
      easy_close=True,
      footer=None,
    )
    ui.modal_show(m)
  data_all.set(data)
  apix_class.set(apix)
  image_size.set(nx)
 
@reactive.effect
@reactive.event(params_orig, data_all, input.ignore_blank, input.sort_abundance)
def get_displayed_class_images():
  req(params_orig() is not None)
  req(data_all() is not None)
  data = data_all()
  n = len(data)
  images = [data[i] for i in range(n)]
 
  df = params_orig()
  abundance = compute.get_class_abundance(df, n)

  display_seq_all = np.arange(n, dtype=int)
  if input.sort_abundance():
    display_seq_all = np.argsort(abundance)[::-1]

  if input.ignore_blank():
    included = []
    for i in range(n):
      image = images[display_seq_all[i]]
      if np.max(image)>np.min(image):
        included.append(display_seq_all[i])
    images = [images[i] for i in included]
  else:
    included = display_seq_all
  image_labels = [f"{i+1}: {abundance[i]:,d}" for i in included]
   
  displayed_class_ids.set(included)
  displayed_class_labels.set(image_labels)
  displayed_class_images.set(images)

@reactive.effect
@reactive.event(input.upload_params)
def get_params_from_upload():
  req(input.input_mode_params()=="upload")
  fileinfo = input.upload_params()
  param_file = fileinfo[0]['datapath']
  if len(fileinfo)==2:
    cs_pass_through_file = fileinfo[1]['datapath']
    assert cs_pass_through_file.endswith(".cs")
  else:
    cs_pass_through_file = None
  try:
    tmp_params = compute.get_class2d_params_from_file(param_file, cs_pass_through_file)
  except:
    tmp_params = None
  params_orig.set(tmp_params)

  if params_orig() is not None:
    apix = compute.get_pixel_size(params_orig(), attrs=["blob/psize_A", "rlnImagePixelSize"])
    if apix:
      ui.update_numeric("apix_particle", value=apix)
  else:
    ui.update_numeric("apix_particle", value=0)
    m = ui.modal(
      f"failed to parse the upload class2D parameters from {fileinfo[0]['name']}",
      title="File upload error",
      easy_close=True,
      footer=None,
    )
    ui.modal_show(m)

@reactive.effect
@reactive.event(input.url_params)
def get_params_from_url():
  req(input.input_mode_params()=="url")
  url = input.url_params()
  try:
    tmp_params = compute.get_class2d_params_from_url(url)
  except:
    tmp_params = None
  params_orig.set(tmp_params)

  if params_orig() is not None:
    apix = compute.get_pixel_size(params_orig(), attrs=["blob/psize_A", "rlnImagePixelSize"])
    if apix:
      ui.update_numeric("apix_particle", value=apix)
  else:
    ui.update_numeric("apix_particle", value=0)
    m = ui.modal(
      f"failed to download class2D parameters from {input.url_params()}",
      title="File download error",
      easy_close=True,
      footer=None,
    )
    ui.modal_show(m)

@reactive.effect
@reactive.event(params_orig)
def get_apix_micrograph_auto():
  apix = compute.get_pixel_size(params_orig(), return_source=True)
  if apix is None:
      apix = input.apix_particle()
      source = "particle pixel size"
  else:
      apix, source = apix
  apix_micrograph_auto.set(apix)
  apix_micrograph_auto_source.set(source)
  ui.update_numeric("apix_micrograph", value=apix)

@reactive.effect
@reactive.event(params_orig, input.apix_micrograph)
def update_particle_locations():
  req(params_orig() is not None)
  req(input.apix_micrograph())
  tmp = compute.update_particle_locations(params=params_orig(), apix_micrograph=input.apix_micrograph())
  params_work.set(tmp)

@reactive.effect
@reactive.event(selected_classes, params_work)
def get_selected_helices():
  req(params_work() is not None)
  class_indices = [displayed_class_ids()[i] for i in selected_classes()]
  helices = compute.select_class(params=params_work(), class_indices=class_indices)
  selected_helices.set(helices)

@reactive.effect
@reactive.event(selected_helices)
def get_selected_helix_lengths():
  req(input.apix_particle())
  req(image_size())
  if len(selected_helices()):
    filement_lengths = compute.get_filament_length(helices=selected_helices(), particle_box_length=input.apix_particle()*image_size())
    selected_helix_lengths.set(filement_lengths)
  else:
    selected_helix_lengths.set([])

@reactive.effect
@reactive.event(selected_helices, input.min_len, input.max_len)
def select_helices_by_length():
  if len(selected_helices())==0:
    retained_helices_by_length.set([])
  elif input.min_len()==0 and input.max_len()<=0:
    retained_helices_by_length.set(selected_helices())
  else:
    helices_retained, n_ptcls = compute.select_helices_by_length(helices=selected_helices(), lengths=selected_helix_lengths(), min_len=input.min_len(), max_len=input.max_len())
    retained_helices_by_length.set(helices_retained)

@reactive.effect
@reactive.event(retained_helices_by_length)
def get_pair_lengths():
  if len(retained_helices_by_length()):
    dists = compute.compute_pair_distances(helices=retained_helices_by_length())
    pair_distances.set(dists)
  else:
    pair_distances.set([])

@reactive.effect
@reactive.event(input.min_len)
def _():
  ui.update_numeric("max_len", min=input.min_len())
  if 0 < input.max_len() < input.min_len():
    ui.update_numeric("max_len", value=-1)

@reactive.effect
@reactive.event(input.max_len)
def _():
  if input.max_len()>0:
    ui.update_numeric("min_len", max=input.max_len())
  if input.min_len() >= input.max_len():
    ui.update_numeric("min_len", value=0)
