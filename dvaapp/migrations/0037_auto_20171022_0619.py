# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-10-22 06:19
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dvaapp', '0036_deepmodel_parent'),
    ]

    operations = [
        migrations.AddField(
            model_name='frame',
            name='event',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.TEvent'),
        ),
        migrations.AddField(
            model_name='segment',
            name='event',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='dvaapp.TEvent'),
        ),
    ]
