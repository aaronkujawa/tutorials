import os
import sys
import shutil
import unittest
import logging
from glob import glob
from xmlrpc.client import INVALID_METHOD_PARAMS
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin, saveNode
from slicer.util import *
import time
import json
import numpy as np

#
# Full_Brain_Parcellation
#

class Full_Brain_Parcellation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Full Brain Parcellation"
    self.parent.categories = ["Segmentation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Aaron Kujawa (King's College London)"]
    self.parent.helpText = """
This scripted loadable module runs a scripted CLI which runs a docker container to perform Full Brain Parcellation 
inference of on T1 images. In the UI one has to select the model (paths to the nnU-Net trained model folders and the 
structure labels can be modified in <Extension Folder>/Resources/model_info.json), input image(s), and output 
segmentation.
"""
    self.parent.acknowledgementText = """
This file is based on a file developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. which was partially funded by NIH grant 3P41RR013218-12S1.
"""

    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

#
# Register sample data sets in Sample Data module
#

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  from pathlib import Path
  module_path = os.path.dirname(__file__)
  iconsPath = os.path.join(module_path, 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.

  # small T1 example image
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='Full_Brain_Parcellation',
    sampleName='small_T1',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'Full_Brain_Parcellation1.png'),
    # Download URL and target file name
    uris=Path(os.path.join(module_path, "Testing", "example_data", "T1", "leadppmila_0412420130410_0000.nrrd")).as_uri(),
    fileNames='smallT1.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:d5f2a5b0917eaf284ac9d681aa0ee0f66d33d16d51bd43eb7eea96d6dd57a97d',
    # This node name will be used when the data set is loaded
    nodeNames='small_T1'
  )

#
# Full_Brain_ParcellationWidget
#

class Full_Brain_ParcellationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/Full_Brain_Parcellation.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = Full_Brain_ParcellationLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector_channel1.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.inputSelector_channel2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.affineregCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.ui.brainextrCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.comboBox.connect("currentIndexChanged(const QString &)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.ui.inputSelector_channel1.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.inputSelector_channel2.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_channel2"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("outputSegmentation"))
    self.ui.affineregCheckBox.checked = (self._parameterNode.GetParameter("AffineReg") == "true")
    self.ui.brainextrCheckBox.checked = (self._parameterNode.GetParameter("BrainExtract") == "true")


    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("outputSegmentation"):
      self.ui.applyButton.toolTip = "Compute output volume"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes"
      self.ui.applyButton.enabled = False

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector_channel1.currentNodeID)
    self._parameterNode.SetNodeReferenceID("InputVolume_channel2", self.ui.inputSelector_channel2.currentNodeID)
    self._parameterNode.SetNodeReferenceID("outputSegmentation", self.ui.outputSelector.currentNodeID)
    self._parameterNode.SetParameter("AffineReg", "true" if self.ui.affineregCheckBox.checked else "false")
    self._parameterNode.SetParameter("BrainExtract", "true" if self.ui.brainextrCheckBox.checked else "false")


    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:

      # assemble input channel nodes
      input_node_list = [n for n in [self.ui.inputSelector_channel1.currentNode(), self.ui.inputSelector_channel2.currentNode()] if n]
      print("Passing " + str(len(input_node_list)) + " input channels")

      # Compute output
      model_key = self.ui.comboBox.currentText
      print("model_key = ", model_key)

      modelInfo = json.loads(self._parameterNode.GetParameter("ModelInfo"))[model_key]
      print("modelInfo = ", modelInfo)

      self.logic.process(input_node_list,
                         self.ui.outputSelector.currentNode(),
                         self.ui.affineregCheckBox.checked,
                         self.ui.brainextrCheckBox.checked,
                         modelInfo)

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()


#
# Full_Brain_ParcellationLogic
#

class Full_Brain_ParcellationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("ModelInfo"):
      module_path = os.path.dirname(__file__)
      with open(os.path.join(module_path, "../../../data/dynunet_trained_models", "model_info.json")) as jsonfile:
        modelInfo = json.load(jsonfile)
        parameterNode.SetParameter("ModelInfo", json.dumps(modelInfo))

    if not parameterNode.GetParameter("AffineReg"):
      parameterNode.SetParameter("AffineReg", "true")

    if not parameterNode.GetParameter("BrainExtract"):
      parameterNode.SetParameter("BrainExtract", "true")

  
  def run_external_inference(self,
                             input_volume_list,
                             output_segmentation,
                             do_affinereg,
                             do_brainextr,
                             modelInfo,
                             ):

    self.start_time = time.time()

    module_path = os.path.dirname(__file__)
    tmp_path_in = os.path.join(module_path, "tmp_in")
    tmp_path_out = os.path.join(module_path, "tmp_out")
    ext_inference_script_path = os.path.join(module_path, "../../../dynunet_pipeline/inference.py")
    model_folds_dir = os.path.join(module_path, "../../../data/dynunet_trained_models")
    datalist_path = os.path.join(module_path, "../../../data/config")

    # get information from modelInfo (read from model_info.json)
    registration_template_path = modelInfo["registration_template_path"]
    task_id = modelInfo["task_id"]
    expr_name = modelInfo["expr_name"]
    fold = modelInfo["fold"]
    checkpoint = modelInfo["checkpoint"]

    with open(os.path.join(datalist_path, f"dataset_task{task_id}.json")) as jsonfile:
      label_dict = json.load(jsonfile)['labels']

    if os.path.isdir(tmp_path_in):
      shutil.rmtree(tmp_path_in)
    if os.path.isdir(tmp_path_out):
      shutil.rmtree(tmp_path_out)
    os.makedirs(tmp_path_in, exist_ok=True)
    os.makedirs(tmp_path_out, exist_ok=True)
        
    # save image to nifti file that can be read by the docker
    for channel, input_volume in enumerate(input_volume_list):
      img_tmp_path = os.path.join(tmp_path_in, "img_tmp_"+"{:04d}".format(channel)+".nii.gz")
      saveNode(input_volume, img_tmp_path)
    
    # provide paths (they must be absolute paths)
    test_images_folder = os.path.realpath(tmp_path_in)
    out_folder = os.path.realpath(tmp_path_out)
    print(f"{registration_template_path=}")
    registration_template_path = os.path.realpath(os.path.join(module_path, "../../../", registration_template_path))

    # check if all paths exist
    print(f"{registration_template_path=}")
    assert(os.path.isdir(test_images_folder)), f"Is not a folder: {test_images_folder} ..."
    assert(os.path.isdir(out_folder)), f"Is not a folder: {out_folder} ..."
    assert(os.path.isfile(registration_template_path)), f"Is not a file: {registration_template_path}"

    cmd = f"""python {ext_inference_script_path} \
    -datalist_path {datalist_path} \
    -model_folds_dir {model_folds_dir} \
    -test_files_dir {test_images_folder}  \
    -val_output_dir {out_folder} \
    -fold {fold} \
    -expr_name {expr_name} \
    -task_id {task_id} \
    -checkpoint {checkpoint} \
    -no-tta_val """

    if do_affinereg:
      cmd += f" -registration_template_path {registration_template_path} "

    if do_brainextr:
      cmd += " -bet "

    print(cmd)
        
    # run scripted CLI
    self.progressBar = slicer.util.createProgressDialog(None, 0, 100)
    
    def onProcessingStatusUpdate(cliNode, event):
      #print("Got a %s from a %s" % (event, cliNode.GetClassName()))
      if cliNode.IsA('vtkMRMLCommandLineModuleNode') and not cliNode.GetStatus()==cliNode.Completed:
        self.progressBar.setValue(cliNode.GetProgress())
        #print("Status is %s" % cliNode.GetStatusString())
      if cliNode.GetStatus() & cliNode.Completed:
        if cliNode.GetStatus() & cliNode.ErrorsMask:
          # error
          errorText = cliNode.GetErrorText()
          print("CLI execution failed: " + errorText)
        else:
          # success
          self.progressBar.setValue(100)
          # get path to output file created by external inference process
          out_files = glob(os.path.join(tmp_path_out, "**", "*.nii.gz"), recursive=True)
          tmp_file_path_out = out_files[0]

          # load the file as a labelVolumeNode
          loadedLabelVolumeNode = slicer.util.loadLabelVolume(tmp_file_path_out)

          # convert the labels in the labelVolumeNode into segments of the output segmentation node
          segmentation = output_segmentation.GetSegmentation()
          segmentation.RemoveAllSegments()

          slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(loadedLabelVolumeNode, output_segmentation)

          # remove the labelVolumeNode
          slicer.mrmlScene.RemoveNode(loadedLabelVolumeNode)

        # rename the segments
          # which structures were predicted by the neural network (without background (0))
          predictedStructureVals = list(set(np.unique(slicer.util.arrayFromVolume(loadedLabelVolumeNode)))-set([0]))

          for seg_idx in range(segmentation.GetNumberOfSegments()):
            segment = segmentation.GetNthSegment(seg_idx)
            orig_idx = int(predictedStructureVals[seg_idx])
            segment.SetName(label_dict[str(orig_idx)])

          #self.progressBar.close()

          # render in 3D
          output_segmentation.CreateClosedSurfaceRepresentation()

          print("CLI execution succeeded after {} seconds".format(int(time.time()-self.start_time)))

    parameters = {"command": cmd}
    
    cliNode = slicer.cli.run(slicer.modules.run_command, None, parameters=parameters)
    cliNode.AddObserver('ModifiedEvent', onProcessingStatusUpdate)


  def process(self, inputVolume_list, outputSegmentation, do_affinereg, do_brainextr, modelInfo):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be segmented
    :param outputSegmentation: segmentation result
    :param do_affinereg: whether to register input volumes to template image
    :param do_brainextr: whether to perform brain extraction on input images
    """

    for i in range(len(inputVolume_list)):
      if not inputVolume_list[i]:
        raise ValueError("Channel " + str(i+1) + " input volume is invalid")
    if not outputSegmentation:
      raise ValueError("Output volume is invalid")

    startTime = time.time()
    logging.info('Processing started')

    self.run_external_inference(inputVolume_list, outputSegmentation, do_affinereg, do_brainextr, modelInfo)

    stopTime = time.time()
    logging.info(f'Python script completed in {stopTime-startTime:.2f} seconds')

#
# Full_Brain_ParcellationTest
#

class Full_Brain_ParcellationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_Full_Brain_Parcellation1()

  # def test_Full_Brain_Parcellation1(self):
  #   """ Ideally you should have several levels of tests.  At the lowest level
  #   tests should exercise the functionality of the logic with different inputs
  #   (both valid and invalid).  At higher levels your tests should emulate the
  #   way the user would interact with your code and confirm that it still works
  #   the way you intended.
  #   One of the most important features of the tests is that it should alert other
  #   developers when their changes will have an impact on the behavior of your
  #   module.  For example, if a developer removes a feature that you depend on,
  #   your test should break so they know that the feature is needed.
  #   """
  #
  #   self.delayDisplay("Starting the test")
  #
  #   # Get/create input data
  #   import SampleData
  #   registerSampleData()
  #
  #   parameterNode = slicer.modules.Full_Brain_ParcellationWidget._parameterNode
  #   modelInfo = json.loads(parameterNode.GetParameter("ModelInfo"))["T1 : full parcellation w/o prior (Task2010)"]
  #
  #   inputVolume = SampleData.downloadSample('small_T1')
  #   self.delayDisplay('Loaded test data set')
  #
  #   outputSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
  #
  #   do_affinereg = (parameterNode.GetParameter("AffineReg") == "true")
  #   do_brainextr = (parameterNode.GetParameter("BrainExtract") == "true")
  #
  #   logic = Full_Brain_ParcellationLogic()
  #   logic.process([inputVolume], outputSegmentation, do_affinereg, do_brainextr, modelInfo)
  #
  #   self.delayDisplay('Test passed')


  def test_Full_Brain_Parcellation1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data
    import SampleData
    registerSampleData()

    parameterNode = slicer.modules.Full_Brain_ParcellationWidget._parameterNode
    modelInfo = json.loads(parameterNode.GetParameter("ModelInfo"))["T1 : full parcellation with prior, w/o BET (Task2120)"]

    module_path = os.path.dirname(__file__)
    priorTemplatePath = os.path.join(module_path, "../../../", "data/test_images/Task2120_regnobetprimix/imagesTs_fold0/regnobetprimix_000003000002_0000.nii.gz")
    PriorTemplate = slicer.util.loadVolume(priorTemplatePath)
    inputVolume = [SampleData.downloadSample('small_T1'), PriorTemplate]
    self.delayDisplay('Loaded test data set')

    outputSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

    do_affinereg = True
    do_brainextr = False

    logic = Full_Brain_ParcellationLogic()
    logic.process(inputVolume, outputSegmentation, do_affinereg, do_brainextr, modelInfo)

    self.delayDisplay('Test passed')