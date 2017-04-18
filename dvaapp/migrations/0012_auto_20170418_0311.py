# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2017-04-18 03:11
from __future__ import unicode_literals

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dvaapp', '0011_auto_20170417_0909'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='clusters',
            name='index_entries',
        ),
        migrations.RemoveField(
            model_name='clusters',
            name='video',
        ),
        migrations.AddField(
            model_name='clusters',
            name='excluded_index_entries_pk',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=[], size=None),
        ),
        migrations.AddField(
            model_name='clusters',
            name='included_index_entries_pk',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=[], size=None),
        ),
        migrations.AddField(
            model_name='clusters',
            name='model_file_name',
            field=models.CharField(default='', max_length=200),
        ),
        migrations.AddField(
            model_name='clusters',
            name='pca_file_name',
            field=models.CharField(default='', max_length=200),
        ),
        migrations.AddField(
            model_name='clusters',
            name='train_fraction',
            field=models.FloatField(default=0.8),
        ),
        migrations.AlterField(
            model_name='clusters',
            name='cluster_count',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='clusters',
            name='components',
            field=models.IntegerField(default=64),
        ),
    ]