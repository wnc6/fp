class MimicID:
    """
    A class to represent a Mimic ID, which is a unique identifier for a subject, study, and DICOM file.
    
    Attributes:
    - subject_id (str): The unique identifier for the subject.
    - study_id (str): The unique identifier for the study.
    - dicom_id (str): The unique identifier for the DICOM file.
    """

    # Initialize the MimicID object with subject_id, study_id, and dicom_id
    def __init__(self, subject_id, study_id, dicom_id):
        """
        Constructs all the necessary attributes for the MimicID object.
        
        Parameters:
        - subject_id (str): The unique identifier for the subject.
        - study_id (str): The unique identifier for the study.
        - dicom_id (str): The unique identifier for the DICOM file.
        """
        self.subject_id = str(subject_id) # Convert to string to ensure type consistency
        self.study_id = str(study_id) # Convert to string to ensure type consistency
        self.dicom_id = str(dicom_id) # Convert to string to ensure type consistency

    def __str__(self):
        """
        Returns a string representation of the MimicID object in the format: p{subject_id}_s{study_id}_{dicom_id}.
        
        Returns:
        - str: A string representation of the MimicID object.
        """
        return f"p{self.subject_id}_s{self.study_id}_{self.dicom_id}"

    @staticmethod
    def get_study_id(mimic_id: str):
        """
        Extracts and returns the study ID from a given Mimic ID string.
        
        Parameters:
        - mimic_id (str): The Mimic ID string from which to extract the study ID.
        
        Returns:
        - str: The extracted study ID.
        """
        return mimic_id.split('_')[1][1:] # Split the string by underscore, take the second part, and remove the leading 's'
