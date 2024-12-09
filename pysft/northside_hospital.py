import syft as sy
from syft.service.user.user import UserCreate, ServiceRole
import pandas as pd

dataset_path = 'dataset/subjects_subset_1.csv'

def create_dataset_assets(path, name, description) -> sy.Asset:
    '''
    Assumption is made the the input path leads to a CSV formatted dataset with substantial rows such that
    100 of the rows can be partitioned into a mock dataset without significantly decreasing the real dataset's 
    size
    '''
    real_df = pd.read_csv(path)

    asset = sy.Asset(
        name=name,
        description=description,
        data=real_df,
        mock=real_df,
    )
    return asset

if __name__ == '__main__':
    domain = sy.orchestra.launch(
        name='northside_hospital',
        reset=True,
        port=8093,
        server_side_type='high',
    )

    admin = domain.login(email="info@openmined.org", password="changethis")
    admin.settings.allow_guest_signup(enable=False)

    admin.users.create(
        name='Gopesh Singal',
        email='gsingal3@gatech.edu',
        password='testing123',
        role=ServiceRole.DATA_SCIENTIST
    )

    asset_subj = create_dataset_assets(path=dataset_path, name='Subjects', description='Personal information for patients')
    dataset = sy.Dataset(
        name='Hospital information', 
        description='Information regarding patients in hospital care for CS 6220',
        asset_list=[asset_subj]
    )

    admin.upload_dataset(dataset)