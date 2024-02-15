from django.contrib import admin
from anime.models import Genre, Anime, Episode, User, Review

@admin.register(Anime)
class AnimeAdmin(admin.ModelAdmin):
    filter_horizontal = ['genre']

# Register your models here.
admin.site.register(Genre)
admin.site.register(Episode)
admin.site.register(User)
admin.site.register(Review)


