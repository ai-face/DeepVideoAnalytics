from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
import requests
import glob
import json
from django.views.generic import ListView, DetailView
from django.utils import timezone
from .forms import UploadFileForm, YTVideoForm, AnnotationForm
from dvaapp.models import Video, Frame, DVAPQL, QueryResults, TEvent, IndexEntries, Region, VDNServer, \
    LOPQCodes, Tube,  Segment, FrameLabel, SegmentLabel, \
    VideoLabel, RegionLabel, TubeLabel, Label, ManagementAction, StoredDVAPQL, \
    DeepModel, Retriever, SystemState, QueryRegion, QueryRegionResults
from dva.celery import app
from rest_framework import viewsets, mixins
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticatedOrReadOnly, IsAuthenticated
from django.db.models import Count
import math
from django.db.models import Max
import view_shared
from dvaapp.processing import DVAPQLProcess
from django.contrib.auth.decorators import user_passes_test,login_required
from django.utils.decorators import method_decorator
from django.contrib.auth.mixins import UserPassesTestMixin
from django_celery_results.models import TaskResult
from rest_framework.authtoken.models import Token
import logging
import defaults

try:
    from django.contrib.postgres.search import SearchVector
except ImportError:
    SearchVector = None
    logging.warning("Could not load Postgres full text search")
from examples import EXAMPLES


class LoginRequiredMixin(object):
    @method_decorator(login_required)
    def dispatch(self, *args, **kwargs):
        return super(LoginRequiredMixin, self).dispatch(*args, **kwargs)


def user_check(user):
    return user.is_authenticated or settings.AUTH_DISABLED


def force_user_check(user):
    return user.is_authenticated


class VideoList(UserPassesTestMixin, ListView):
    model = Video
    paginate_by = 100
    template_name = "dvaui/video_list.html"

    def get_context_data(self, **kwargs):
        context = super(VideoList, self).get_context_data(**kwargs)
        context['exports'] = TEvent.objects.all().filter(operation='perform_export')
        return context

    def test_func(self):
        return user_check(self.request.user)


class TEventDetail(UserPassesTestMixin, DetailView):
    model = TEvent
    template_name = "dvaui/tevent_detail.html"

    def get_context_data(self, **kwargs):
        context = super(TEventDetail, self).get_context_data(**kwargs)
        try:
            tr = TaskResult.objects.get(task_id=context['object'].task_id)
        except TaskResult.DoesNotExist:
            context['celery_task'] = None
            pass
        else:
            context['celery_task'] = tr
        return context

    def test_func(self):
        return user_check(self.request.user)


class TEventList(UserPassesTestMixin, ListView):
    model = TEvent
    paginate_by = 500
    template_name = "dvaui/tevent_list.html"

    def get_queryset(self):
        kwargs = {}
        if self.kwargs.get('pk',None):
            kwargs['video_id']=self.kwargs['pk']
        elif self.kwargs.get('process_pk',None):
            kwargs['parent_process_id']=self.kwargs['process_pk']
        if self.kwargs.get('status',None):
            if self.kwargs['status'] == 'running':
                kwargs['duration__lt'] = 0
                kwargs['started'] = True
                kwargs['completed'] = False
                kwargs['errored'] = False
            elif self.kwargs['status'] == 'successful':
                kwargs['completed'] = True
            elif self.kwargs['status'] == 'pending':
                kwargs['duration__lt'] = 0
                kwargs['started'] = False
                kwargs['errored'] = False
            elif self.kwargs['status'] == 'failed':
                kwargs['errored'] = True
        new_context = TEvent.objects.filter(**kwargs).order_by('-created')
        return new_context

    def get_context_data(self, **kwargs):
        view_shared.refresh_task_status()
        context = super(TEventList, self).get_context_data(**kwargs)
        context['header'] = ""
        if self.kwargs.get('pk',None):
            context['video'] = Video.objects.get(pk=self.kwargs['pk'])
            context['header'] = "video/dataset : {}".format(context['video'].name)
        if self.kwargs.get('process_pk',None):
            process_pk = self.kwargs.get('process_pk',None)
            context['header'] = "process : {}".format(process_pk)
        if self.kwargs.get('status',None):
            context['header'] += " with status {}".format(self.kwargs['status'])
        return context

    def test_func(self):
        return user_check(self.request.user)


class VideoDetail(UserPassesTestMixin, DetailView):
    model = Video
    template_name = "dvaui/video_detail.html"


    def get_context_data(self, **kwargs):
        context = super(VideoDetail, self).get_context_data(**kwargs)
        max_frame_index = Frame.objects.all().filter(video=self.object).aggregate(Max('frame_index'))[
            'frame_index__max']
        context['exports'] = TEvent.objects.all().filter(operation='perform_export', video=self.object)
        context['annotation_count'] = Region.objects.all().filter(video=self.object,
                                                                  region_type=Region.ANNOTATION).count()
        context['exportable_annotation_count'] = 0
        context['url'] = '{}{}/video/{}.mp4'.format(settings.MEDIA_URL, self.object.pk, self.object.pk)
        label_list = []
        show_all = self.request.GET.get('show_all_labels', False)
        context['label_list'] = label_list
        delta = 10000
        if context['object'].dataset:
            delta = 500
        if max_frame_index <= delta:
            context['frame_list'] = Frame.objects.all().filter(video=self.object).order_by('frame_index')
            context['detection_list'] = Region.objects.all().filter(video=self.object, region_type=Region.DETECTION)
            context['annotation_list'] = Region.objects.all().filter(video=self.object, region_type=Region.ANNOTATION)
            context['offset'] = 0
            context['limit'] = max_frame_index
        else:
            if self.request.GET.get('frame_index_offset', None) is None:
                offset = 0
            else:
                offset = int(self.request.GET.get('frame_index_offset'))
            limit = offset + delta
            context['offset'] = offset
            context['limit'] = limit
            context['frame_list'] = Frame.objects.all().filter(video=self.object, frame_index__gte=offset,
                                                               frame_index__lte=limit).order_by('frame_index')
            context['detection_list'] = Region.objects.all().filter(video=self.object, frame_index__gte=offset,
                                                                    frame_index__lte=limit,
                                                                    region_type=Region.DETECTION)
            context['annotation_list'] = Region.objects.all().filter(video=self.object, frame_index__gte=offset,
                                                                     frame_index__lte=limit,
                                                                     region_type=Region.ANNOTATION)
            context['frame_index_offsets'] = [(k * delta, (k * delta) + delta) for k in
                                              range(int(math.ceil(max_frame_index / float(delta))))]
        context['frame_first'] = context['frame_list'].first()
        context['frame_last'] = context['frame_list'].last()
        context['pending_tasks'] = TEvent.objects.all().filter(video=self.object,started=False, errored=False).count()
        context['running_tasks'] = TEvent.objects.all().filter(video=self.object,started=True, completed=False, errored=False).count()
        context['successful_tasks'] = TEvent.objects.all().filter(video=self.object,completed=True).count()
        context['errored_tasks'] = TEvent.objects.all().filter(video=self.object,errored=True).count()
        if context['limit'] > max_frame_index:
            context['limit'] = max_frame_index
        context['max_frame_index'] = max_frame_index
        return context

    def test_func(self):
        return user_check(self.request.user)


# class RetrieverDetails(UserPassesTestMixin, DetailView):
#     model = Clusters
#
#     def get_context_data(self, **kwargs):
#         context = super(ClustersDetails, self).get_context_data(**kwargs)
#         context['coarse'] = []
#         context['index_entries'] = [IndexEntries.objects.get(pk=k) for k in self.object.included_index_entries_pk]
#         for k in ClusterCodes.objects.filter(clusters_id=self.object.pk).values('coarse_text').annotate(
#                 count=Count('coarse_text')):
#             context['coarse'].append({'coarse_text': k['coarse_text'].replace(' ', '_'),
#                                       'count': k['count'],
#                                       'first': ClusterCodes.objects.all().filter(clusters_id=self.object.pk,
#                                                                                  coarse_text=k['coarse_text']).first(),
#                                       'last': ClusterCodes.objects.all().filter(clusters_id=self.object.pk,
#                                                                                 coarse_text=k['coarse_text']).last()
#                                       })
#
#         return context
#
#     def test_func(self):
#         return user_check(self.request.user)


class DetectionDetail(UserPassesTestMixin, DetailView):
    model = DeepModel
    template_name = "dvaui/customdetector_detail.html"


    def get_context_data(self, **kwargs):
        context = super(DetectionDetail, self).get_context_data(**kwargs)
        classdist = context['object'].class_distribution.strip()
        context['class_distribution'] = json.loads(classdist) if classdist else {}
        context['phase_1_log'] = []
        context['phase_2_log'] = []
        for k in context['object'].phase_1_log.split('\n')[1:]:
            if k.strip():
                epoch,train_loss,val_loss = k.strip().split(',')
                context['phase_1_log'].append((epoch,round(float(train_loss),2),round(float(val_loss),2)))
        for k in context['object'].phase_2_log.split('\n')[1:]:
            if k.strip():
                epoch,train_loss,val_loss = k.strip().split(',')
                context['phase_2_log'].append((epoch,round(float(train_loss),2),round(float(val_loss),2)))
        return context

    def test_func(self):
        return user_check(self.request.user)


class FrameDetail(UserPassesTestMixin, DetailView):
    model = Frame
    template_name = 'dvaui/frame_detail.html'

    def get_context_data(self, **kwargs):
        context = super(FrameDetail, self).get_context_data(**kwargs)
        context['detection_list'] = Region.objects.all().filter(frame=self.object, region_type=Region.DETECTION)
        context['annotation_list'] = Region.objects.all().filter(frame=self.object, region_type=Region.ANNOTATION)
        context['video'] = self.object.video
        context['url'] = '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL, self.object.video.pk, self.object.frame_index)
        context['previous_frame'] = Frame.objects.filter(video=self.object.video,
                                                         frame_index__lt=self.object.frame_index).order_by(
            '-frame_index')[0:1]
        context['next_frame'] = Frame.objects.filter(video=self.object.video,
                                                     frame_index__gt=self.object.frame_index).order_by('frame_index')[
                                0:1]
        return context

    def test_func(self):
        return user_check(self.request.user)


class SegmentDetail(UserPassesTestMixin, DetailView):
    model = Segment
    template_name = 'dvaui/segment_detail.html'

    def get_context_data(self, **kwargs):
        context = super(SegmentDetail, self).get_context_data(**kwargs)
        context['video'] = self.object.video
        context['frame_list'] = Frame.objects.all().filter(video=self.object.video,segment_index=self.object.segment_index).order_by('frame_index')
        context['region_list'] = Region.objects.all().filter(video=self.object.video,segment_index=self.object.segment_index).order_by('frame_index')
        context['url'] = '{}{}/segments/{}.mp4'.format(settings.MEDIA_URL, self.object.video.pk, self.object.segment_index)
        context['previous_segment_index'] = self.object.segment_index - 1 if self.object.segment_index else None
        context['next_segment_index'] = self.object.segment_index + 1 if (self.object.segment_index+1) < self.object.video.segments else None
        return context

    def test_func(self):
        return user_check(self.request.user)


class VisualSearchList(UserPassesTestMixin, ListView):
    model = DVAPQL
    template_name = "dvaui/query_list.html"

    def test_func(self):
        return user_check(self.request.user)

    def get_queryset(self):
        new_context = DVAPQL.objects.filter(process_type=DVAPQL.QUERY).order_by('-created')
        return new_context


class VisualSearchDetail(UserPassesTestMixin, DetailView):
    model = DVAPQL
    template_name = "dvaui/query_detail.html"

    def get_context_data(self, **kwargs):
        context = super(VisualSearchDetail, self).get_context_data(**kwargs)
        qp = DVAPQLProcess(process=context['object'],media_dir=settings.MEDIA_ROOT)
        qp_context = view_shared.collect(qp)
        context['results'] = qp_context['results'].items()
        context['regions'] = qp_context['regions']
        script = context['object'].script
        script[u'image_data_b64'] = "<excluded>"
        context['plan'] = script
        context['pending_tasks'] = TEvent.objects.all().filter(parent_process=self.object,started=False, errored=False).count()
        context['running_tasks'] = TEvent.objects.all().filter(parent_process=self.object,started=True, completed=False, errored=False).count()
        context['successful_tasks'] = TEvent.objects.all().filter(parent_process=self.object,completed=True).count()
        context['errored_tasks'] = TEvent.objects.all().filter(parent_process=self.object,errored=True).count()
        context['url'] = '{}queries/{}.png'.format(settings.MEDIA_URL, self.object.pk, self.object.pk)
        return context

    def test_func(self):
        return user_check(self.request.user)


class ProcessList(UserPassesTestMixin, ListView):
    model = DVAPQL
    template_name = "dvaui/process_list.html"
    paginate_by = 50

    def get_context_data(self, **kwargs):
        context = super(ProcessList, self).get_context_data(**kwargs)
        return context

    def test_func(self):
        return user_check(self.request.user)

    def get_queryset(self):
        new_context = DVAPQL.objects.filter().order_by('-created')
        return new_context


class ProcessDetail(UserPassesTestMixin, DetailView):
    model = DVAPQL
    template_name = "dvaui/process_detail.html"

    def get_context_data(self, **kwargs):
        context = super(ProcessDetail, self).get_context_data(**kwargs)
        context['json'] = json.dumps(context['object'].script,indent=4)
        context['pending_tasks'] = TEvent.objects.all().filter(parent_process=self.object,started=False, errored=False).count()
        context['running_tasks'] = TEvent.objects.all().filter(parent_process=self.object,started=True, completed=False, errored=False).count()
        context['successful_tasks'] = TEvent.objects.all().filter(parent_process=self.object,completed=True).count()
        context['errored_tasks'] = TEvent.objects.all().filter(parent_process=self.object,errored=True).count()
        return context

    def test_func(self):
        return user_check(self.request.user)


class StoredProcessList(UserPassesTestMixin, ListView):
    model = StoredDVAPQL
    template_name = "dvaui/stored_process_list.html"
    paginate_by = 20
    ordering = "-created"

    def get_context_data(self, **kwargs):
        context = super(StoredProcessList, self).get_context_data(**kwargs)
        context['examples'] = json.dumps(EXAMPLES, indent=None)
        return context

    def test_func(self):
        return user_check(self.request.user)


class StoredProcessDetail(UserPassesTestMixin, DetailView):
    model = StoredDVAPQL
    template_name = "dvaui/stored_process_detail.html"

    def get_context_data(self, **kwargs):
        context = super(StoredProcessDetail, self).get_context_data(**kwargs)
        context['json'] = json.dumps(context['object'].script,indent=4)
        return context

    def test_func(self):
        return user_check(self.request.user)


@user_passes_test(user_check)
def search(request):
    if request.method == 'POST':
        qp = DVAPQLProcess()
        view_shared.create_query_from_request(qp,request)
        qp.launch()
        qp.wait()
        qp_context = view_shared.collect(qp)
        return JsonResponse(data={'task_id': "",
                                  'primary_key': qp.process.pk,
                                  'results': qp_context['results'],
                                  'regions': qp_context['regions'],
                                  'url': '{}queries/{}.png'.format(settings.MEDIA_URL, qp.process.pk)
                                  })


def home(request):
    if request.user.is_authenticated:
        return redirect("app")
    else:
        return render(request, 'home.html', {})


def paper(request):
    return render(request, 'paper.html', {})


@user_passes_test(user_check)
def index(request, query_pk=None, frame_pk=None, detection_pk=None):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        user = request.user if request.user.is_authenticated else None
        if form.is_valid():
            view_shared.handle_uploaded_file(request.FILES['file'], form.cleaned_data['name'], user=user,rate=form.cleaned_data['nth'],
                                 rescale=form.cleaned_data['rescale'] if 'rescale' in form.cleaned_data else 0)
            return redirect('video_list')
        else:
            raise ValueError
    else:
        form = UploadFileForm()
    context = {'form': form}
    context['detectors'] = DeepModel.objects.filter(model_type=DeepModel.DETECTOR)
    context['indexer_retrievers'] = []
    for i in DeepModel.objects.filter(model_type=DeepModel.INDEXER):
        for r in Retriever.objects.all():
            if 'indexer_shasum' in r.source_filters and r.source_filters['indexer_shasum'] == i.shasum and r.last_built:
                context['indexer_retrievers'].append(('{} > {} retriever {} (pk:{})'.format(i.name,
                                                      r.get_algorithm_display(),r.name,r.pk),
                                                      '{}_{}'.format(i.pk,r.pk)))
    if query_pk:
        previous_query = DVAPQL.objects.get(pk=query_pk)
        context['initial_url'] = '{}queries/{}.png'.format(settings.MEDIA_URL, query_pk)
    elif frame_pk:
        frame = Frame.objects.get(pk=frame_pk)
        context['initial_url'] = '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL, frame.video.pk, frame.frame_index)
    elif detection_pk:
        detection = Region.objects.get(pk=detection_pk)
        context['initial_url'] = '{}{}/regions/{}.jpg'.format(settings.MEDIA_URL, detection.video.pk, detection.pk)
    context['frame_count'] = Frame.objects.count()
    context['query_count'] = DVAPQL.objects.filter(process_type=DVAPQL.QUERY).count()
    context['process_count'] = DVAPQL.objects.filter(process_type=DVAPQL.PROCESS).count()
    context['index_entries_count'] = IndexEntries.objects.count()
    context['external_servers_count'] = VDNServer.objects.count()
    context['task_events_count'] = TEvent.objects.count()
    context['pending_tasks'] = TEvent.objects.all().filter(started=False, errored=False).count()
    context['running_tasks'] = TEvent.objects.all().filter(started=True, completed=False, errored=False).count()
    context['successful_tasks'] = TEvent.objects.all().filter(started=True, completed=True).count()
    context['errored_tasks'] = TEvent.objects.all().filter(errored=True).count()
    context['video_count'] = Video.objects.count()
    context['index_entries'] = IndexEntries.objects.all()
    context['region_count'] = Region.objects.all().count()
    context['tube_count'] = Tube.objects.all().count()
    context["videos"] = Video.objects.all().filter()
    context['detector_count'] = DeepModel.objects.filter(model_type=DeepModel.DETECTOR).count()
    context['rate'] = defaults.DEFAULT_RATE
    return render(request, 'dashboard.html', context)


@user_passes_test(user_check)
def assign_video_labels(request):
    if request.method == 'POST':
        video = Video.objects.get(pk=request.POST.get('video_pk'))
        spec = []
        for k in request.POST.get('labels').split(','):
            if k.strip():
                spec.append({
                    'MODEL':'VideoLabel',
                    'spec':{'video_id':video.pk,'label_id':Label.objects.get_or_create(name=k,set="UI")[0].id}
                })
        p = DVAPQLProcess()
        p.create_from_json({
            'process_type':DVAPQL.PROCESS,
            'create':spec,
        },user=request.user if request.user.is_authenticated else None)
        p.launch()
        return redirect('video_detail', pk=video.pk)
    else:
        raise NotImplementedError


@user_passes_test(user_check)
def annotate(request, frame_pk):
    context = {'frame': None, 'detection': None, 'existing': []}
    frame = None
    frame = Frame.objects.get(pk=frame_pk)
    context['frame'] = frame
    context['initial_url'] = '{}{}/frames/{}.jpg'.format(settings.MEDIA_URL, frame.video.pk, frame.frame_index)
    context['previous_frame'] = Frame.objects.filter(video=frame.video, frame_index__lt=frame.frame_index).order_by(
        '-frame_index')[0:1]
    context['next_frame'] = Frame.objects.filter(video=frame.video, frame_index__gt=frame.frame_index).order_by(
        'frame_index')[0:1]
    context['detections'] = Region.objects.filter(frame=frame, region_type=Region.DETECTION)
    for d in Region.objects.filter(frame=frame):
        temp = {
            'x': d.x,
            'y': d.y,
            'h': d.h,
            'w': d.w,
            'pk': d.pk,
            'box_type': "detection" if d.region_type == d.DETECTION else 'annotation',
            'label': d.object_name,
            'full_frame': d.full_frame,
            'detection_pk': None
        }
        context['existing'].append(temp)
    context['existing'] = json.dumps(context['existing'])
    if request.method == 'POST':
        form = AnnotationForm(request.POST)
        if form.is_valid():
            applied_tags = form.cleaned_data['tags'].split(',') if form.cleaned_data['tags'] else []
            view_shared.create_annotation(form, form.cleaned_data['object_name'], applied_tags, frame)
            return JsonResponse({'status': True})
        else:
            raise ValueError, form.errors
    return render(request, 'annotate.html', context)


@user_passes_test(user_check)
def annotate_entire_frame(request, frame_pk):
    frame = Frame.objects.get(pk=frame_pk)
    annotation = None
    if request.method == 'POST':
        if request.POST.get('text').strip() \
                or request.POST.get('metadata').strip() \
                or request.POST.get('object_name', None):
            annotation = Region()
            annotation.region_type = Region.ANNOTATION
            annotation.x = 0
            annotation.y = 0
            annotation.h = 0
            annotation.w = 0
            annotation.full_frame = True
            annotation.text = request.POST.get('text')
            annotation.metadata = request.POST.get('metadata')
            annotation.object_name = request.POST.get('object_name', 'frame_metadata')
            annotation.frame = frame
            annotation.video = frame.video
            annotation.save()
        for label_name in request.POST.get('tags').split(','):
            if label_name.strip():
                if annotation:
                    dl = RegionLabel()
                    dl.video = frame.video
                    dl.frame = frame
                    dl.label = Label.objects.get_or_create(name=label_name,set="UI")[0]
                    dl.region = annotation
                    dl.save()
                else:
                    dl = FrameLabel()
                    dl.video = frame.video
                    dl.frame = frame
                    dl.label = Label.objects.get_or_create(name=label_name,set="UI")[0]
                    dl.save()
    return redirect("frame_detail", pk=frame.pk)


@user_passes_test(user_check)
def yt(request):
    if request.method == 'POST':
        form = YTVideoForm(request.POST, request.FILES)
        user = request.user if request.user.is_authenticated else None
        if form.is_valid():
            rate = form.cleaned_data['nth']
            rescale = form.cleaned_data['rescale'] if 'rescale' in form.cleaned_data else 0
            video = view_shared.handle_video_url(form.cleaned_data['name'], form.cleaned_data['url'], user=user)
            process_spec = {
                'process_type': DVAPQL.PROCESS,
                'tasks': [{'video_id': video.pk,
                          'operation': 'perform_import',
                          'arguments': {'source': "URL",
                                        'next_tasks':[{'video_id': video.pk,
                                                       'operation': 'perform_video_segmentation',
                                                       'arguments': {
                                                           'next_tasks': [
                                                               {'operation': 'perform_video_decode',
                                                                'arguments': {
                                                                    'rate': rate, 'rescale': rescale,
                                                                    'segments_batch_size': defaults.DEFAULT_SEGMENTS_BATCH_SIZE,
                                                                    'next_tasks': defaults.DEFAULT_PROCESSING_PLAN_VIDEO
                                                                }
                                                                }
                                                           ]},
                                                       },]
                                        }
                          },]
                }
            p = DVAPQLProcess()
            p.create_from_json(process_spec, user)
            p.launch()
        else:
            raise ValueError
    else:
        raise NotImplementedError
    return redirect('video_list')


@user_passes_test(user_check)
def segment_by_index(request,video_pk,segment_index):
    segment = Segment.objects.get(video_id=video_pk,segment_index=segment_index)
    return redirect('segment_detail',pk=segment.pk)


@user_passes_test(user_check)
def export_video(request):
    if request.method == 'POST':
        pk = request.POST.get('video_id')
        video = Video.objects.get(pk=pk)
        export_method = request.POST.get('export_method')
        if video:
            if export_method == 's3':
                key = request.POST.get('key')
                bucket = request.POST.get('bucket')
                region = request.POST.get('region','us-east-1')
                process_spec = {'process_type':DVAPQL.PROCESS,
                          'tasks':[
                              {
                                  'video_id':video.pk,
                                  'operation':'perform_export',
                                  'arguments': {'key':key,'bucket':bucket,'region':region,'destination':'S3'}
                              },
                          ]}
            else:
                process_spec = {'process_type':DVAPQL.PROCESS,
                          'tasks':[
                              {
                                  'video_id':video.pk,
                                  'operation':'perform_export',
                                  'arguments':{'destination':'FILE'}
                              },
                          ]
                          }
            p = DVAPQLProcess()
            p.create_from_json(process_spec)
            p.launch()
        return redirect('video_list')
    else:
        raise NotImplementedError


@user_passes_test(user_check)
def coarse_code_detail(request, pk, coarse_code):
    coarse_code = coarse_code.replace('_', ' ')
    context = {
        'code': coarse_code,
        'objects': LOPQCodes.objects.all().filter(coarse_text=coarse_code, clusters_id=pk)
    }
    return render(request, 'coarse_code_details.html', context)


@user_passes_test(user_check)
def status(request):
    context = {}
    context['logs'] = []
    for fname in glob.glob('logs/*.log'):
        context['logs'].append((fname,file(fname).read()))
    return render(request, 'status.html', context)


@user_passes_test(user_check)
def management(request):
    timeout = 1.0
    context = {
        'timeout':timeout,
        'actions':ManagementAction.objects.all(),
        'state':SystemState.objects.all().order_by('-created')[:100]
    }
    if request.method == 'POST':
        op = request.POST.get("op","")
        host_name = request.POST.get("host_name","").strip()
        queue_name = request.POST.get("queue_name","").strip()
        if op =="list_workers":
            context["queues"] = app.control.inspect(timeout=timeout).active_queues()
        elif op == "list":
            t = app.send_task('manage_host', args=[op, ], exchange='qmanager')
            t.wait(timeout=timeout)
        elif op == "gpuinfo":
            t = app.send_task('manage_host', args=[op, ], exchange='qmanager')
            t.wait(timeout=timeout)
        elif op == "launch":
            t = app.send_task('manage_host', args=[op,host_name,queue_name],exchange='qmanager')
            t.wait(timeout=timeout)
    return render(request, 'management.html', context)


@user_passes_test(user_check)
def training(request):
    context = {}
    context["videos"] = Video.objects.all().filter()
    context["detectors"] = DeepModel.objects.filter(model_type=DeepModel.DETECTOR)
    return render(request, 'training.html', context)


@user_passes_test(user_check)
def textsearch(request):
    context = {'results': {}, "videos": Video.objects.all().filter()}
    q = request.GET.get('q')
    if q:
        offset = int(request.GET.get('offset',0))
        delta = int(request.GET.get('delta',25))
        limit = offset + delta
        context['q'] = q
        context['next'] = limit
        context['delta'] = delta
        context['offset'] = offset
        context['limit'] = limit
        if request.GET.get('regions'):
            context['results']['regions_meta'] = Region.objects.filter(text__search=q)[offset:limit]
            context['results']['regions_name'] = Region.objects.filter(object_name__search=q)[offset:limit]
        if request.GET.get('frames'):
            context['results']['frames_name'] = Frame.objects.filter(name__search=q)[offset:limit]
            context['results']['frames_subdir'] = Frame.objects.filter(subdir__search=q)[offset:limit]
        if request.GET.get('labels'):
            context['results']['labels'] = Label.objects.filter(name__search=q)[offset:limit]
    return render(request, 'textsearch.html', context)


@user_passes_test(user_check)
def retrievers(request):
    context = {}
    context['algorithms'] = {k.name for k in DeepModel.objects.filter(model_type=DeepModel.INDEXER)}
    context['index_entries'] = IndexEntries.objects.all()
    context['retrievers'] = Retriever.objects.all()
    return render(request, 'retrievers.html', context)


@user_passes_test(user_check)
def create_retriever(request):
    if request.method == 'POST':
        spec = {}
        if request.POST.get('retriever_type') == Retriever.LOPQ:
            v = request.POST.get('v')
            m = request.POST.get('m')
            components = request.POST.get('components')
            sub = request.POST.get('sub')
            spec['name'] = request.POST.get('name')
            spec['algorithm'] = Retriever.LOPQ
            args = {}
            args['components']= components
            args['sub']= sub
            args['m']= m
            args['v']= v
            spec['arguments'] = args
            if request.POST.get('source_filters',None):
                spec['source_filters'] = json.loads(request.POST.get('source_filter','{}'))
            else:
                spec['source_filters'] = {'indexer_shasum':DeepModel.objects.get(name=request.POST.get('algorithm'),
                                                                                 model_type=DeepModel.INDEXER).shasum}
            next_tasks = [{'operation': "perform_retriever_creation",'arguments': {'retriever_pk':'__pk__'},},]
        elif request.POST.get('retriever_type') == Retriever.EXACT:
            spec['name'] = request.POST.get('name')
            spec['last_built'] = '__timezone.now__'
            spec['source_filters'] = json.loads(request.POST.get('source_filters', '{}'))
            spec['algorithm'] = Retriever.EXACT
            next_tasks = []
        else:
            raise ValueError
        if spec:
            p = DVAPQLProcess()
            p.create_from_json(j={"process_type": DVAPQL.PROCESS,
                                  "create":[{'MODEL':'Retriever','spec':spec,'tasks': next_tasks}],
                                  }, user=request.user if request.user.is_authenticated else None)
            p.launch()
    return redirect('retrievers')


@user_passes_test(user_check)
def submit_process(request):
    if request.method == 'POST':
        process_pk = request.POST.get('process_pk',None)
        if process_pk is None:
            p = DVAPQLProcess()
            p.create_from_json(j=json.loads(request.POST.get('script')), user=request.user if request.user.is_authenticated else None)
            p.launch()
        else:
            p = DVAPQLProcess(process=DVAPQL.objects.get(pk=process_pk))
            p.launch()
        return redirect("process_detail",pk=p.process.pk)


@user_passes_test(user_check)
def validate_process(request):
    if request.method == 'POST':
        p = DVAPQLProcess()
        p.create_from_json(j=json.loads(request.POST.get('script')), user=request.user if request.user.is_authenticated else None)
        p.validate()
    return redirect("process_detail",pk=p.process.pk)


@user_passes_test(user_check)
def delete_object(request):
    if request.method == 'POST':
        pk = request.POST.get('pk')
        if request.POST.get('object_type') == 'annotation':
            annotation = Region.objects.get(pk=pk)
            if annotation.region_type == Region.ANNOTATION:
                annotation.delete()
    return JsonResponse({'status': True})


@user_passes_test(force_user_check)
def security(request):
    context = {}
    context['username'] = request.user.username
    token, created = Token.objects.get_or_create(user=request.user if request.user.is_authenticated else None)
    context['token'] = token
    return render(request, 'security.html', context=context)


@user_passes_test(force_user_check)
def expire_token(request):
    # TODO Check if this is correct
    if request.method == 'POST':
        if request.POST.get('expire',False):
            token, created = Token.objects.get_or_create(user=request.user if request.user.is_authenticated else None)
            if not created:
                token.delete()
    return redirect("security")


@user_passes_test(user_check)
def import_dataset(request):
    if request.method == 'POST':
        url = request.POST.get('dataset_url')
        server = VDNServer.objects.get(pk=request.POST.get('server_pk'))
        user = request.user if request.user.is_authenticated else None
        cached_response = server.last_response_datasets[int(request.POST.get('dindex'))]
        view_shared.import_vdn_dataset_url(server, url, user, cached_response)
    else:
        raise NotImplementedError
    return redirect('video_list')


@user_passes_test(user_check)
def import_detector(request):
    if request.method == 'POST':
        url = request.POST.get('detector_url')
        server = VDNServer.objects.get(pk=request.POST.get('server_pk'))
        user = request.user if request.user.is_authenticated else None
        cached_response = server.last_response_detectors[int(request.POST.get('dindex'))]
        view_shared.import_vdn_detector_url(server, url, user, cached_response)
    else:
        raise NotImplementedError
    return redirect('models')


@user_passes_test(user_check)
def import_s3(request):
    if request.method == 'POST':
        keys = request.POST.get('key')
        region = request.POST.get('region')
        bucket = request.POST.get('bucket')
        rate = request.POST.get('rate',defaults.DEFAULT_RATE)
        rescale = request.POST.get('rescale',defaults.DEFAULT_RESCALE)
        user = request.user if request.user.is_authenticated else None
        create = []
        for key in keys.strip().split('\n'):
            tasks =[]
            key = key.strip()
            if key:
                extract_task = {'arguments': {'rate': rate, 'rescale': rescale,
                                              'next_tasks': defaults.DEFAULT_PROCESSING_PLAN_DATASET},
                                 'operation': 'perform_dataset_extraction'}
                segment_decode_task = {'operation': 'perform_video_segmentation',
                                        'arguments': {
                                            'next_tasks': [
                                                {'operation': 'perform_video_decode',
                                                 'arguments': {
                                                     'rate': rate, 'rescale': rescale,
                                                     'segments_batch_size':defaults.DEFAULT_SEGMENTS_BATCH_SIZE,
                                                     'next_tasks': defaults.DEFAULT_PROCESSING_PLAN_VIDEO
                                                }
                                            }
                                            ]},
                                        }
                if key.endswith('.dva_export.zip'):
                    next_tasks = []
                elif key.endswith('.zip'):
                    next_tasks = [extract_task,]
                else:
                    next_tasks = [segment_decode_task,]
                tasks.append({'video_id':'__pk__',
                              'operation':'perform_import',
                              'arguments':{'key':key,
                                           'bucket':bucket,
                                           'region':region,
                                           'source':'S3',
                                           'next_tasks':next_tasks}
                              })
                create.append({'MODEL': 'Video',
                               'spec': {'uploader_id': user.pk if user else None,
                                        'name': "pending S3 import {} s3://{}/{}".format(region, bucket, key)},
                               'tasks': tasks
                               })
        process_spec = {'process_type': DVAPQL.PROCESS,
                        'create':create}
        p = DVAPQLProcess()
        p.create_from_json(process_spec,user)
        p.launch()
    else:
        raise NotImplementedError
    return redirect('video_list')


@user_passes_test(user_check)
def external(request):
    if request.method == 'POST':
        pk = request.POST.get('server_pk')
        try:
            view_shared.pull_vdn_list(pk)
        except:
            pass
    context = {
        'servers': VDNServer.objects.all(),
        'available_datasets': {server: server.last_response_datasets for server in VDNServer.objects.all()},
        'available_detectors': {server: server.last_response_detectors for server in VDNServer.objects.all()},
    }
    return render(request, 'external_data.html', context)


@user_passes_test(user_check)
def retry_task(request):
    pk = request.POST.get('pk')
    event = TEvent.objects.get(pk=int(pk))
    spec = {
        'process_type':DVAPQL.PROCESS,
        'tasks':[
            {
                'operation':event.operation,
                'arguments':event.arguments
            }
        ]
    }
    p = DVAPQLProcess()
    p.create_from_json(spec)
    p.launch()
    return redirect('/processes/')


@user_passes_test(user_check)
def delete_video(request):
    if request.user.is_staff: # currently only staff can delete
        video_pk = request.POST.get('video_id')
        view_shared.delete_video_object(video_pk,request.user)
        return redirect('video_list')
    else:
        return redirect('accounts/login/')


@user_passes_test(user_check)
def rename_video(request):
    if request.user.is_staff: # currently only staff can rename
        video_pk = request.POST.get('video_id')
        name = request.POST.get('name')
        v = Video.objects.get(pk=int(video_pk))
        v.name = name
        v.save()
        return redirect('video_list')
    else:
        return redirect('accounts/login/')


@user_passes_test(user_check)
def models(request):
    context = {
        'visual_index_list': DeepModel.objects.filter(model_type=DeepModel.INDEXER),
        'index_entries': IndexEntries.objects.all(),
        'analyzers': DeepModel.objects.filter(model_type=DeepModel.ANALYZER),
        "videos": Video.objects.all(),
        "region_types": Region.REGION_TYPES,
        "detectors": DeepModel.objects.filter(model_type=DeepModel.DETECTOR),
        "deep_models": DeepModel.objects.all()
    }
    return render(request, 'models.html', context)


@user_passes_test(user_check)
def index_video(request):
    if request.method == 'POST':
        filters = {
            'region_type__in': request.POST.getlist('region_type__in', []),
            'w__gte': int(request.POST.get('w__gte')),
            'h__gte': int(request.POST.get('h__gte'))
         }
        for optional_key in ['text__contains', 'object_name__contains', 'object_name']:
            if request.POST.get(optional_key, None):
                filters[optional_key] = request.POST.get(optional_key)
        for optional_key in ['h__lte', 'w__lte']:
            if request.POST.get(optional_key, None):
                filters[optional_key] = int(request.POST.get(optional_key))
        args = {'filters':filters,'index':request.POST.get('visual_index_name')}
        p = DVAPQLProcess()
        spec = {
            'process_type':DVAPQL.PROCESS,
            'tasks':[
                {
                    'operation':'perform_indexing',
                    'arguments':args,
                    'video_id':request.POST.get('video_id')
                }
            ]
        }
        user = request.user if request.user.is_authenticated else None
        p.create_from_json(spec,user)
        p.launch()
        redirect('process_detail',pk=p.process.pk)
    else:
        raise ValueError


@user_passes_test(user_check)
def detect_objects(request):
    if request.method == 'POST':
        detector_pk = request.POST.get('detector_pk')
        video_pk = request.POST.get('video_pk')
        p = DVAPQLProcess()
        p.create_from_json(j={
            "process_type":DVAPQL.PROCESS,
            "tasks":[{'operation':"perform_detection",
                      'arguments':{'detector_pk': int(detector_pk),'detector':"custom"},
                      'video_id':video_pk}]
        },user=request.user if request.user.is_authenticated else None)
        p.launch()
        return redirect('process_detail',pk=p.process.pk)
    else:
        raise ValueError


@user_passes_test(user_check)
def train_detector(request):
    if request.method == 'POST':
        args = request.POST.get('args')
        args = json.loads(args) if args.strip() else {}
        args['name'] = request.POST.get('name')
        args['labels'] = [k.strip() for k in request.POST.get('labels').split(',') if k.strip()]
        args['object_names'] = [k.strip() for k in request.POST.get('object_names').split(',') if k.strip()]
        args['excluded_videos'] = request.POST.getlist('excluded_videos')
        args['detector_pk'] = '__pk__'
        p = DVAPQLProcess()
        p.create_from_json(j={
            "process_type":DVAPQL.PROCESS,
            "create":[
                {
                    'MODEL':'Detector',
                    'spec':{
                        'name':args['name'],
                        'arguments':json.dumps(args),
                        'algorithm':'yolo'
                    },
                    "tasks":[
                        {'operation':"perform_detector_training",
                         'arguments':args,
                         }]
                }
            ]
        },user=request.user if request.user.is_authenticated else None)
        p.launch()
        return redirect('process_detail', pk=p.process.pk)
    # elif request.POST.get('action') == 'estimate':
    #     args = request.POST.get('args')
    #     args = json.loads(args) if args.strip() else {}
    #     args['name'] = request.POST.get('name')
    #     args['labels'] = [k.strip() for k in request.POST.get('labels').split(',') if k.strip()]
    #     args['object_names'] = [k.strip() for k in request.POST.get('object_names').split(',') if k.strip()]
    #     args['excluded_videos'] = request.POST.getlist('excluded_videos')
    #     labels = set(args['labels']) if 'labels' in args else set()
    #     object_names = set(args['object_names']) if 'object_names' in args else set()
    #     class_distribution, class_names, rboxes, rboxes_set,
    #     frames, i_class_names = create_detector_dataset(object_names, labels)
    #     context["estimate"] = {
    #         'args':args,
    #         'class_distribution':class_distribution,
    #         'class_names':class_names,
    #         'rboxes':rboxes,
    #         'rboxes_set':rboxes_set,
    #         'frames':frames,
    #         'i_class_names':i_class_names
    #     }
    else:
        raise ValueError
