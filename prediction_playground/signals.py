from django.db.models.signals import pre_delete, post_delete
from django.dispatch import receiver

from prediction_playground.models import SKPipeline


@receiver(pre_delete, sender=SKPipeline)
def handlePreDelete(sender, **kwargs):
    instance = kwargs['instance']
    # Store the OrderLines as a property of the object
    if instance.previous_version is not None:
        instance.old_parent_id = instance.previous_version.id
    else:
        instance.old_parent_id = None
    instance.old_child_ids = list(instance.skpipeline_set.all().values_list("id", flat=True))

@receiver(post_delete, sender=SKPipeline)
def handlePostDelete(sender, **kwargs):
    instance = kwargs['instance']
    for child in SKPipeline.objects.filter(id__in=instance.old_child_ids):
        if instance.old_parent_id is not None:
            child.previous_version = SKPipeline.objects.get(id=instance.old_parent_id)
        else:
            child.previous_version = None
        child.save()


