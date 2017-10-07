import os
import logging
import time
import glob
import subprocess
import calendar
import sys
from fabric.api import task, local, lcd
import json
import shutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/fab.log',
                    filemode='a')


@task
def shell():
    """
    start a local django shell
    """
    local('python manage.py shell')


@task
def local_static():
    """
    Collect static
    """
    local('python manage.py collectstatic')


@task
def migrate():
    """
    Make migrations and migrate database
    """
    local('python manage.py makemigrations')
    local('python manage.py migrate')


@task
def server():
    """
    Start server locally
    """
    local("python manage.py runserver")


@task
def pull_private():
    """
    Pull from private repo
    """
    local('aws s3 cp s3://aub3config/.netrc /root/.netrc')
    local('git clone https://github.com/AKSHAYUBHAT/DeepVideoAnalyticsDemo')
    local('mv DeepVideoAnalyticsDemo dvap')
    local('rm /root/.netrc')
    with lcd('dvap'):
        local('./setup_private.sh')


@task
def start_container(container_type):
    """
    Start container with queues launched as specified in environment
    """
    if container_type == 'worker':
        time.sleep(30)  # To avoid race condition where worker starts before migration is finished
        init_fs()
        init_models()
        launch_workers_and_scheduler_from_environment(block_on_manager=True)
    elif container_type == 'server':
        init_fs()
        init_server()
        init_models()
        launch_workers_and_scheduler_from_environment()
        launch_server_from_environment()
    else:
        raise ValueError, "invalid container_type = {}".format(container_type)


@task
def clean():
    """
    Reset database, migrate, clear media folder, and (only on my dev machine) kill workers/clear all queues.
    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    from dvaapp import queuing
    if sys.platform == 'darwin':
        for qname in set(queuing.TASK_NAMES_TO_QUEUE.values()):
            try:
                local('rabbitmqadmin purge queue name={}'.format(qname))
            except:
                logging.warning("coudnt clear queue {}".format(qname))
    # TODO: wait for Celery bug fix https://github.com/celery/celery/issues/3620
    # local('celery amqp exchange.delete broadcast_tasks')
    migrate()
    local('python manage.py flush --no-input')
    migrate()
    local("rm -rf {}/*".format(settings.MEDIA_ROOT))
    local("mkdir {}/queries".format(settings.MEDIA_ROOT))
    if sys.platform == 'darwin':
        local("rm logs/*.log")
        try:
            local("ps auxww | grep 'celery -A dva' | awk '{print $2}' | xargs kill -9")
        except:
            pass
    init_fs()
    init_server()
    init_models()
    if sys.platform == 'darwin':
        superu()


@task
def restart_queues():
    """
    Kill all workers and launch them again

    """
    kill()
    launch()


@task
def kill():
    try:
        local("ps auxww | grep 'celery -A dva * ' | awk '{print $2}' | xargs kill -9")
    except:
        pass


@task
def ci():
    """
    Perform Continuous Integration testing using Travis

    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    import base64
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaui.view_shared import handle_uploaded_file, pull_vdn_list \
        , import_vdn_dataset_url
    from dvaapp.models import Video, TEvent, VDNServer, DVAPQL, Retriever, DeepModel
    from django.conf import settings
    from dvaapp.processing import DVAPQLProcess
    from dvaapp.tasks import perform_dataset_extraction, perform_indexing, perform_export, perform_import, \
        perform_retriever_creation, perform_detection, \
        perform_video_segmentation, perform_transformation
    for fname in glob.glob('tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name, False)
    if sys.platform != 'darwin':
        for fname in glob.glob('tests/*.mp4'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
            handle_uploaded_file(f, name, False)
        for fname in glob.glob('tests/*.zip'):
            name = fname.split('/')[-1].split('.')[0]
            f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
            handle_uploaded_file(f, name)
    for i, v in enumerate(Video.objects.all()):
        if v.dataset:
            arguments = {'sync': True}
            perform_dataset_extraction(TEvent.objects.create(video=v, arguments=arguments).pk)
        else:
            arguments = {'sync': True}
            perform_video_segmentation(TEvent.objects.create(video=v, arguments=arguments).pk)
        arguments = {'index': 'inception', 'target': 'frames'}
        perform_indexing(TEvent.objects.create(video=v, arguments=arguments).pk)
        if i == 0:  # save travis time by just running detection on first video
            # face_mtcnn
            arguments = {'detector': 'face'}
            dt = TEvent.objects.create(video=v, arguments=arguments)
            perform_detection(dt.pk)
            arguments = {'filters': {'event_id': dt.pk}, }
            perform_transformation(TEvent.objects.create(video=v, arguments=arguments).pk)
            # coco_mobilenet
            arguments = {'detector': 'coco'}
            dt = TEvent.objects.create(video=v, arguments=arguments)
            perform_detection(dt.pk)
            arguments = {'filters': {'event_id': dt.pk}, }
            perform_transformation(TEvent.objects.create(video=v, arguments=arguments).pk)
            # inception on crops from detector
            arguments = {'index': 'inception', 'target': 'regions',
                         'filters': {'event_id': dt.pk, 'w__gte': 50, 'h__gte': 50}}
            perform_indexing(TEvent.objects.create(video=v, arguments=arguments).pk)
            # assign_open_images_text_tags_by_id(TEvent.objects.create(video=v).pk)
        temp = TEvent.objects.create(video=v, arguments={'destination': "FILE"})
        perform_export(temp.pk)
        temp.refresh_from_db()
        fname = temp.arguments['file_name']
        f = SimpleUploadedFile(fname, file("{}/exports/{}".format(settings.MEDIA_ROOT, fname)).read(),
                               content_type="application/zip")
        vimported = handle_uploaded_file(f, fname)
        perform_import(TEvent.objects.create(video=vimported, arguments={"source": "LOCAL"}).pk)
    dc = Retriever()
    args = {}
    args['components'] = 32
    args['m'] = 8
    args['v'] = 8
    args['sub'] = 64
    dc.algorithm = Retriever.LOPQ
    dc.source_filters = {'indexer_shasum': DeepModel.objects.get(name="inception",model_type=DeepModel.INDEXER).shasum}
    dc.arguments = args
    dc.save()
    clustering_task = TEvent()
    clustering_task.arguments = {'retriever_pk': dc.pk}
    clustering_task.operation = 'perform_retriever_creation'
    clustering_task.save()
    perform_retriever_creation(clustering_task.pk)
    query_dict = {
        'process_type': DVAPQL.QUERY,
        'image_data_b64': base64.encodestring(file('tests/query.png').read()),
        'tasks': [
            {
                'operation': 'perform_indexing',
                'arguments': {
                    'index': 'inception',
                    'target': 'query',
                    'next_tasks': [
                        {'operation': 'perform_retrieval',
                         'arguments': {'count': 20, 'retriever_pk': Retriever.objects.get(name='inception').pk}
                         }
                    ]
                }

            }

        ]
    }
    launch_workers_and_scheduler_from_environment()
    qp = DVAPQLProcess()
    qp.create_from_json(query_dict)
    qp.launch()
    qp.wait()
    server, datasets, detectors = pull_vdn_list(1)
    for k in datasets:
        if k['name'] == 'MSCOCO_Sample_500':
            print 'FOUND MSCOCO SAMPLE'
            import_vdn_dataset_url(VDNServer.objects.get(pk=1), k['url'], None, k)


@task
def ci_search():
    """
    Perform Continuous Integration testing using Travis for search queries
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    import base64
    from dvaapp.models import DVAPQL, Retriever, QueryResults
    from dvaapp.processing import DVAPQLProcess
    launch_workers_and_scheduler_from_environment()
    query_dict = {
        'process_type': DVAPQL.QUERY,
        'image_data_b64': base64.encodestring(file('tests/query.png').read()),
        'tasks': [
            {
                'operation': 'perform_indexing',
                'arguments': {
                    'index': 'inception',
                    'target': 'query',
                    'next_tasks': [
                        {'operation': 'perform_retrieval',
                         'arguments': {'count': 15, 'retriever_pk': Retriever.objects.get(name='inception',
                                                                                          algorithm=Retriever.EXACT).pk}
                         }
                    ]
                }

            },
            {
                'operation': 'perform_detection',
                'arguments': {
                    'detector': 'coco',
                    'target': 'query',
                }

            }

        ]
    }
    qp = DVAPQLProcess()
    qp.create_from_json(query_dict)
    qp.launch()
    qp.wait(timeout=360)
    print QueryResults.objects.count()


@task
def ci_face():
    """
    Perform Continuous Integration testing using Travis for face detection / indexing

    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Video, TEvent
    from dvaapp.tasks import perform_indexing
    for i, v in enumerate(Video.objects.all()):
        if i == 0:  # save travis time by just running detection on first video
            args = {
                'filter': {'object_name__startswith': 'MTCNN_face'},
                'index': 'facenet',
                'target': 'regions'}
            perform_indexing(TEvent.objects.create(video=v, arguments=args).pk)


@task
def quick():
    """
    Clear and launch for testing on dev machine

    """
    clean()
    superu()
    test()
    launch()


@task
def superu(username="akshay", email="akshay@test.com", password="super"):
    """
    Create a superuser
    :param username: 
    :param email: 
    :param password: 
    """
    local(
        'echo "from django.contrib.auth.models import User; User.objects.create_superuser(\'{}\', \'{}\', \'{}\')" | python manage.py shell'.format(
            username, email, password))


@task
def launch():
    """
    Launch workers on dev machine by adding environment variables

    """
    envars = ['LAUNCH_BY_NAME_indexer_inception', 'LAUNCH_BY_NAME_indexer_facenet',
              'LAUNCH_BY_NAME_retriever_inception', 'LAUNCH_BY_NAME_retriever_facenet',
              'LAUNCH_BY_NAME_detector_coco', 'LAUNCH_BY_NAME_detector_face', 'LAUNCH_BY_NAME_analyzer_tagger',
              'LAUNCH_Q_qclusterer', 'LAUNCH_Q_qextract','LAUNCH_SCHEDULER']
    for k in envars:
        os.environ[k] = "1"
    launch_workers_and_scheduler_from_environment(False)


def launch_workers_and_scheduler_from_environment(block_on_manager=False):
    """
    Launch workers and scheduler as specified in the environment variables.
    Only one scheduler should be launched per deployment.

    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import DeepModel, Retriever
    from dvaapp import queuing
    for k in os.environ:
        if k.startswith('LAUNCH_BY_NAME_'):
            qtype, model_name = k.split('_')[-2:]
            env_vars = ""
            if qtype == 'indexer':
                dm = DeepModel.objects.get(name=model_name,model_type=DeepModel.INDEXER)
                queue_name = 'q_indexer_{}'.format(dm.pk)
                env_vars = "PYTORCH_MODE=1 " if dm.mode == dm.PYTORCH else env_vars
                env_vars = "CAFFE_MODE=1 " if dm.mode == dm.CAFFE else env_vars
                env_vars = "MXNET_MODE=1 " if dm.mode == dm.MXNET else env_vars
            elif qtype == 'retriever':
                dm = Retriever.objects.get(name=model_name)
                queue_name = 'q_retriever_{}'.format(dm.pk)
            elif qtype == 'detector':
                dm = DeepModel.objects.get(name=model_name,model_type=DeepModel.DETECTOR)
                queue_name = 'q_detector_{}'.format(dm.pk)
                env_vars = "PYTORCH_MODE=1 " if dm.mode == dm.PYTORCH else env_vars
                env_vars = "CAFFE_MODE=1 " if dm.mode == dm.CAFFE else env_vars
                env_vars = "MXNET_MODE=1 " if dm.mode == dm.MXNET else env_vars
            elif qtype == 'analyzer':
                dm = DeepModel.objects.get(name=model_name,model_type=DeepModel.ANALYZER)
                queue_name = 'q_analyzer_{}'.format(dm.pk)
                env_vars = "PYTORCH_MODE=1 " if dm.mode == dm.PYTORCH else env_vars
                env_vars = "CAFFE_MODE=1 " if dm.mode == dm.CAFFE else env_vars
                env_vars = "MXNET_MODE=1 " if dm.mode == dm.MXNET else env_vars
            else:
                raise ValueError, k
            command = '{}fab startq:{} &'.format(env_vars, queue_name)
            logging.info("'{}' for {}".format(command, k))
            local(command)
        elif k.startswith('LAUNCH_Q_') and k != 'LAUNCH_Q_{}'.format(queuing.Q_MANAGER):
            if k.strip() == 'LAUNCH_Q_qextract':
                queue_name = k.split('_')[-1]
                local('fab startq:{},{} &'.format(queue_name, os.environ['LAUNCH_Q_qextract']))
            else:
                queue_name = k.split('_')[-1]
                local('fab startq:{} &'.format(queue_name))
    if os.environ.get("LAUNCH_SCHEDULER", False):
        # Should be launched only once per deployment
        local('fab start_scheduler &')
    if block_on_manager:  # the container process waits on the manager
        local('fab startq:{}'.format(queuing.Q_MANAGER))
    else:
        local('fab startq:{} &'.format(queuing.Q_MANAGER))


def launch_server_from_environment():
    """
    Launch django development server or NGINX server as specified in environment variable

    """
    if 'LAUNCH_SERVER' in os.environ:
        local('python manage.py runserver 0.0.0.0:8000')
    elif 'LAUNCH_SERVER_NGINX' in os.environ:
        local('chmod 0777 -R /tmp')
        try:
            local("mv configs/nginx.conf /etc/nginx/")
        except:
            print "warning assuming that the config was already moved"
            pass
        if 'ENABLE_BASICAUTH' in os.environ:
            try:
                local("mv configs/nginx-app_password.conf /etc/nginx/sites-available/default")
            except:
                print "warning assuming that the config was already moved"
                pass
        else:
            try:
                local("mv configs/nginx-app.conf /etc/nginx/sites-available/default")
            except:
                print "warning assuming that the config was already moved"
                pass
        try:
            local("mv configs/supervisor-app.conf /etc/supervisor/conf.d/")
        except:
            print "warning assuming that the config was already moved"
            pass
        local("python manage.py collectstatic --no-input")
        local("chmod 0777 -R dva/staticfiles/")
        # local("chmod 0777 -R dva/media/")
        local('supervisord -n')


@task
def init_scheduler():
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django_celery_beat.models import PeriodicTask,IntervalSchedule
    di,created = IntervalSchedule.objects.get_or_create(every=os.environ.get('REFRESH_MINUTES',3),period=IntervalSchedule.MINUTES)
    _ = PeriodicTask.objects.get_or_create(name="monitoring",task="monitor_system",interval=di,queue='qscheduler')


@task
def init_server():
    """
    Initialize server database by adding default VDN server and DVAPQL templates
 
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Video, VDNServer, StoredDVAPQL
    if StoredDVAPQL.objects.count() == 0:
        for fname in glob.glob('configs/templates/*.json'):
            StoredDVAPQL.objects.create(name=fname,
                                        process_type=StoredDVAPQL.PROCESS,
                                        script=json.loads(file(fname).read()))
    if not ('DISABLE_VDN' in os.environ):
        if VDNServer.objects.count() == 0:
            servers = json.loads(file('configs/vdn_servers.json').read())
            for s in servers:
                server = VDNServer()
                server.url = s['url']
                server.name = s['name']
                server.last_response_datasets = s['last_response_datasets']
                server.last_response_detectors = s['last_response_detectors']
                server.save()
    if 'TEST' in os.environ and Video.objects.count() == 0:
        test()


def download_model(root_dir, model_type_dir_name, model_dir_name, model_json):
    """
    Download model to filesystem
    """
    filename, url = model_json['filename'], model_json['url']
    model_type_dir = "{}/{}/".format(root_dir, model_type_dir_name)
    if not os.path.isdir(model_type_dir):
        os.mkdir(model_type_dir)
    model_dir = "{}/{}/{}".format(root_dir, model_type_dir_name, model_dir_name)
    with lcd(model_type_dir):
        if not os.path.isdir(model_dir):
# <<<<<<< HEAD
#             os.mkdir(model_dir)
#             #if sys.platform == 'darwin':
#             local("cd {} && cp /home/tom/ai/DeepVideoAnalytics.shared/models/{} .".format(model_dir_name, filename))
#             #else:
#             #    local("cd {} && wget --quiet {}".format(model_dir_name, url))
#             if 'additional_files' in model_json:
#                 for m in model_json["additional_files"]:
#                     url = m['url']
#                     filename = m['filename']
#                     #if sys.platform == 'darwin':
#                     local("cd {} && cp /home/tom/ai/DeepVideoAnalytics.shared/models/{} .".format(model_dir_name,
#                                                                                                       filename))
#                     #else:
#                     #    local("cd {} && wget --quiet {}".format(model_dir_name, url))
# =======
            try:
                os.mkdir(model_dir)
            except:
                pass
            else: # On the shared FS the which creates the DIR gets to download
                if True : # sys.platform == 'darwin':
                    local("cd {} && cp /home/tom/ai/DeepVideoAnalytics.shared/models/{} .".format(model_dir_name, filename))
                else:
                    local("cd {} && wget --quiet {}".format(model_dir_name, url))
                if 'additional_files' in model_json:
                    for m in model_json["additional_files"]:
                        url = m['url']
                        filename = m['filename']
                        if True : #sys.platform == 'darwin':
                            local("cd {} && cp /home/tom/ai/DeepVideoAnalytics.shared/models/{} .".format(model_dir_name,
                                                                                                          filename))
                        else:
                            local("cd {} && wget --quiet {}".format(model_dir_name, url))
# >>>>>>> origin/master


@task
def init_models():
    """
    Initialize default models in database specified in models.json,
    and download models to filesystem. Models are downloaded even if the database
    entries exist , but files doe not  since the worker might not be running in shared filesystem model.
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    from django.utils import timezone
    from dvaapp.models import DeepModel, Retriever
    with open("configs/models.json") as modelfile:
        models = json.load(modelfile)
    for m in models:
        if m['model_type'] == "detector":
            dm, _ = DeepModel.objects.get_or_create(name=m['name'],
                                                   algorithm=m['algorithm'],
                                                   mode=m['mode'],
                                                   model_filename=m.get("filename", ""),
                                                   detector_type=m.get("detector_type", ""),
                                                   class_index_to_string=m.get("class_index_to_string", {}),
                                                   model_type=DeepModel.DETECTOR
                                                   )
            if m['url']:
                download_model(settings.MEDIA_ROOT, "models", dm.pk, m)
        if m['model_type'] == "indexer":
            dm, created = DeepModel.objects.get_or_create(name=m['name'], mode=m['mode'], shasum=m['shasum'],model_type=DeepModel.INDEXER)
            if created:
                dr, dcreated = Retriever.objects.get_or_create(name=m['name'], source_filters={'indexer_shasum': dm.shasum})
                if dcreated:
                    dr.last_built = timezone.now()
                    dr.save()
            if m['url']:
                download_model(settings.MEDIA_ROOT, "models", dm.pk, m)
        if m['model_type'] == "analyzer":
            dm, _ = DeepModel.objects.get_or_create(name=m['name'], mode=m['mode'],model_type=DeepModel.ANALYZER)
            if m['url']:
                download_model(settings.MEDIA_ROOT, "models", dm.pk, m)


@task
def init_fs():
    """
    Initialize filesystem by creating directories
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    for create_dirname in ['queries', 'exports', 'external', 'retrievers', 'ingest']:
        if not os.path.isdir("{}/{}".format(settings.MEDIA_ROOT, create_dirname)):
            try:
                os.mkdir("{}/{}".format(settings.MEDIA_ROOT, create_dirname))
            except:
                pass


@task
def startq(queue_name, conc=3):
    """
    Start worker to handle a queue, Usage: fab startq:indexer
    Concurrency is set to 1 but you can edit code to change.
    :param conc:conccurency only for extractor

    """
    import django, os
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp import queuing
    mute = '--without-gossip --without-mingle --without-heartbeat' if 'CELERY_MUTE' in os.environ else ''
    if queue_name == queuing.Q_MANAGER:
        command = 'celery -A dva worker -l info {} -c 1 -Q qmanager -n manager.%h -f logs/qmanager.log'.format(mute)
    elif queue_name == queuing.Q_EXTRACTOR:
        command = 'celery -A dva worker -l info {} -c {} -Q {} -n {}.%h -f logs/{}.log'.format(mute, max(int(conc), 2),
                                                                                               queue_name, queue_name,
                                                                                               queue_name)
        # TODO: worker fails due to
        # https://github.com/celery/celery/issues/3620
    else:
        command = 'celery -A dva worker -l info {} -P solo -c {} -Q {} -n {}.%h -f logs/{}.log'.format(mute, 1,
                                                                                                       queue_name,
                                                                                                       queue_name,
                                                                                                       queue_name)
    logging.info(command)
    os.system(command)


@task
def test():
    """
    Run tests by launching tasks

    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaui.view_shared import handle_uploaded_file, handle_video_url
    for fname in glob.glob('tests/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="video/mp4")
        handle_uploaded_file(f, name)
    for fname in glob.glob('tests/*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        handle_uploaded_file(f, name)


@task
def backup():
    """
    Take a backup, backups are store as a single zip file in backups/ folder

    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    try:
        os.mkdir('backups')
    except:
        pass
    media_dir = settings.MEDIA_ROOT
    db = settings.DATABASES.values()[0]
    pg = '/Users/aub3/PostgreSQL/pg96/bin/pg_dump' if sys.platform == 'darwin' else 'pg_dump'
    with open('{}/postgres.dump'.format(media_dir), 'w') as dumpfile:
        dump = subprocess.Popen([pg, '--clean', '--dbname',
                                 'postgresql://{}:{}@{}:5432/{}'.format(db['USER'], db['PASSWORD'], db['HOST'],
                                                                        db['NAME'])], cwd=media_dir, stdout=dumpfile)
        dump.communicate()
    print dump.returncode
    current_path = os.path.abspath(os.path.dirname(__file__))
    command = ['zip', '-r', '{}/backups/backup_{}.zip'.format(current_path, calendar.timegm(time.gmtime())), '.']
    print ' '.join(command)
    zipper = subprocess.Popen(command, cwd=media_dir)
    zipper.communicate()
    os.remove('{}/postgres.dump'.format(media_dir))
    print zipper.returncode


@task
def restore(path):
    """
    Restore a backup using path provided. Note that arugment are provided in following format. fab restore:backups/backup_1.zip
    :param path:

    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.conf import settings
    media_dir = settings.MEDIA_ROOT
    current_path = os.path.abspath(os.path.dirname(__file__))
    command = ['unzip', '-o', '{}'.format(os.path.join(current_path, path))]
    print ' '.join(command)
    zipper = subprocess.Popen(command, cwd=media_dir)
    zipper.communicate()
    db = settings.DATABASES.values()[0]
    pg = '/Users/aub3/PostgreSQL/pg96/bin/psql' if sys.platform == 'darwin' else 'psql'
    with open('{}/postgres.dump'.format(media_dir)) as dumpfile:
        dump = subprocess.Popen(
            [pg, '--dbname',
             'postgresql://{}:{}@{}:5432/{}'.format(db['USER'], db['PASSWORD'], db['HOST'], db['NAME'])],
            cwd=media_dir, stdin=dumpfile)
        dump.communicate()
    print dump.returncode
    os.remove('{}/postgres.dump'.format(media_dir))
    print zipper.returncode


def setup_django():
    """
    setup django
    :return:
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()


@task
def train_yolo(start_pk):
    """
    Train a yolo model specified in a TaskEvent.
    This is necessary to ensure that the Tensorflow process exits and releases the allocated GPU memory.
    :param start_pk: TEvent PK with information about lauching the training task

    """
    setup_django()
    from django.conf import settings
    from dvaapp.models import Region, Frame, DeepModel, TEvent
    from dvaui.view_shared import create_detector_dataset
    from dvalib.yolo import trainer
    start = TEvent.objects.get(pk=start_pk)
    args = start.arguments
    labels = set(args['labels']) if 'labels' in args else set()
    object_names = set(args['object_names']) if 'object_names' in args else set()
    detector = DeepModel.objects.get(pk=args['detector_pk'])
    detector.create_directory()
    args['root_dir'] = "{}/detectors/{}/".format(settings.MEDIA_ROOT, detector.pk)
    args['base_model'] = "{}/detectors/yolo/yolo.h5"
    class_distribution, class_names, rboxes, rboxes_set, frames, i_class_names = create_detector_dataset(object_names,
                                                                                                         labels)
    images, boxes = [], []
    path_to_f = {}
    for k, f in frames.iteritems():
        path = "{}/{}/frames/{}.jpg".format(settings.MEDIA_ROOT, f.video_id, f.frame_index)
        path_to_f[path] = f
        images.append(path)
        boxes.append(rboxes[k])
        # print k,rboxes[k]
    with open("{}/input.json".format(args['root_dir']), 'w') as input_data:
        json.dump({'boxes': boxes,
                   'images': images,
                   'args': args,
                   'class_names': class_names.items(),
                   'class_distribution': class_distribution.items()},
                  input_data)
    detector.boxes_count = sum([len(k) for k in boxes])
    detector.frames_count = len(images)
    detector.classes_count = len(class_names)
    detector.save()
    args['class_names'] = i_class_names
    train_task = trainer.YOLOTrainer(boxes=boxes, images=images, args=args)
    train_task.train()
    detector.phase_1_log = file("{}/phase_1.log".format(args['root_dir'])).read()
    detector.phase_2_log = file("{}/phase_2.log".format(args['root_dir'])).read()
    detector.class_distribution = json.dumps(class_distribution.items())
    detector.class_names = json.dumps(class_names.items())
    detector.trained = True
    detector.save()
    results = train_task.predict()
    bulk_regions = []
    for path, box_class, score, top, left, bottom, right in results:
        r = Region()
        r.region_type = r.ANNOTATION
        r.confidence = int(100.0 * score)
        r.object_name = "YOLO_{}_{}".format(detector.pk, box_class)
        r.y = top
        r.x = left
        r.w = right - left
        r.h = bottom - top
        r.frame_id = path_to_f[path].pk
        r.video_id = path_to_f[path].video_id
        bulk_regions.append(r)
    Region.objects.bulk_create(bulk_regions, batch_size=1000)
    folder_name = "{}/detectors/{}".format(settings.MEDIA_ROOT, detector.pk)
    file_name = '{}/exports/{}.dva_detector.zip'.format(settings.MEDIA_ROOT, detector.pk)
    zipper = subprocess.Popen(['zip', file_name, '-r', '.'], cwd=folder_name)
    zipper.wait()
    return 0


@task
def temp_import_detector(path="/Users/aub3/tempd"):
    """
    Test importing detectors
    """
    setup_django()
    import json
    from django.conf import settings
    from dvaapp.models import DeepModel
    d = DeepModel()
    with open("{}/input.json".format(path)) as infile:
        data = json.load(infile)
    d.name = "test detector"
    d.class_names = json.dumps(data['class_names'])
    d.phase_1_log = file("{}/phase_1.log".format(path)).read
    d.phase_2_log = file("{}/phase_2.log".format(path)).read
    d.frames_count = 500
    d.boxes_count = 500
    d.detector_type = d.YOLO
    d.model_type = DeepModel.DETECTOR
    d.class_distribution = json.dumps(data['class_names'])
    d.save()
    d.create_directory()
    shutil.copy("{}/phase_2_best.h5".format(path), "{}/detectors/{}/phase_2_best.h5".format(settings.MEDIA_ROOT, d.pk))


@task
def qt():
    """
    Add short videos/datasets and launch default tasks for quick testing
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.core.files.uploadedfile import SimpleUploadedFile
    from dvaui.view_shared import handle_uploaded_file
    for fname in glob.glob('tests/ci/*.mp4'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/mp4")
        _ = handle_uploaded_file(f, name)
    for fname in glob.glob('tests/example*.zip'):
        name = fname.split('/')[-1].split('.')[0]
        f = SimpleUploadedFile(fname, file(fname).read(), content_type="application/zip")
        _ = handle_uploaded_file(f, name)


@task
def qt_lopq():
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.models import Retriever, DeepModel,TEvent,DVAPQL
    from dvaapp import processing
    spec = {
        'process_type':DVAPQL.PROCESS,
        'create':[{
            'MODEL':'Retriever',
            'spec':{
                'algorithm':Retriever.LOPQ,
                'arguments':{'components': 32, 'm': 8, 'v': 8, 'sub': 128},
                'source_filters':{'indexer_shasum': DeepModel.objects.get(name="inception",model_type=DeepModel.INDEXER).shasum}
            },
            'tasks':[
                {
                    'operation':'perform_retriever_creation',
                    'arguments':{'retriever_pk': '__pk__'}
                }
            ]
        },]
    }
    p = processing.DVAPQLProcess()
    p.create_from_json(j=spec,user=None)
    p.launch()
    p.wait()


@task
def submit(path):
    """
    Submit a DVAPQL process to run
    :param path:
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from dvaapp.processing import DVAPQLProcess
    with open(path) as f:
        j = json.load(f)
    p = DVAPQLProcess()
    p.create_from_json(j)
    p.launch()
    print "launched Process with id {} ".format(p.process.pk)


@task
def start_scheduler():
    """
    Start celery-beat scheduler using django database as source for tasks.

    """
    local("celery -A dva beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler -f logs/beat.log")


@task
def store_token_for_testing():
    """
    Generate & store authentication token for superuser (akshay) to test REST API.
    """
    import django
    sys.path.append(os.path.dirname(__file__))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dva.settings")
    django.setup()
    from django.contrib.auth.models import User
    from rest_framework.authtoken.models import Token
    try:
        u = User.objects.get(username="akshay")
    except:
        superu()
        u = User.objects.get(username="akshay")
    token, _ = Token.objects.get_or_create(user=User.objects.get(username=u))
    with open('creds.json', 'w') as creds:
        creds.write(json.dumps({'token': token.key}))


@task
def test_api(port=80):
    """
    test REST API for CORS config by submitting a DVAPQL query to /api endpoint
    """
    import requests
    if not os.path.isfile('creds.json'):
        store_token_for_testing()
    token = json.loads(file('creds.json').read())['token']
    headers = {'Authorization': 'Token {}'.format(token)}
    r = requests.post("http://localhost:{}/api/queries/".format(port),
                      data={'script': file('dvaapp/test_scripts/url.json').read()},
                      headers=headers)
    print r.status_code


@task
def capture_stream(url="https://www.youtube.com/watch?v=vpm16w3ik0g"):
    """
    Test capturing live video feed (experimental)
    """
    command = 'livestreamer --player-continuous-http --player-no-close ' \
              '"{}" best -O --yes-run-as-root | ' \
              'ffmpeg -re -i - -c:v libx264 -c:a aac -ac 1 -strict -2 -crf 18 ' \
              '-profile:v baseline -maxrate 3000k -bufsize 1835k -pix_fmt yuv420p ' \
              '-flags -global_header -f segment -segment_time 0.1 "%d.mp4"'.format(url)
    if raw_input(
            "This code uses os.system and is a huge security risk if url is malicious shell string. Type yes to confirm>>") == "yes":
        print command
        os.system(command)
