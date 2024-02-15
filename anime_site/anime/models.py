from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

# Create your models here.
class Genre(models.Model):
    title = models.CharField(max_length=30)

    def __str__(self):
        return self.title


class Anime(models.Model):
    anime_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=100)
    description = models.TextField()
    poster = models.ImageField(upload_to="anime_posters")
    studio = models.CharField(max_length=200)
    rating = models.DecimalField(max_digits=2, decimal_places=1, validators=[MaxValueValidator(5), MinValueValidator(0)])
    start_release_date = models.DateField(auto_now_add=True)
    end_release_date = models.DateField(auto_now_add=True)
    genre = models.ManyToManyField(Genre, related_name="animes")

    def __str__(self):
        return self.title
    
class Episode(models.Model):
    title = models.CharField(max_length=200)
    number = models.PositiveIntegerField()
    duration = models.PositiveIntegerField()
    release_date = models.DateField(auto_now_add=True)
    anime = models.ForeignKey(Anime, on_delete=models.CASCADE)

    def __str__(self) -> str:
        return self.title
    
class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    email = models.EmailField()
    registration_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.username
    
class Review(models.Model):
    review_id = models.AutoField(primary_key=True, auto_created=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    anime = models.ForeignKey(Anime, on_delete=models.CASCADE)
    text = models.TextField()
    rating = models.IntegerField(validators=[MaxValueValidator(5), MinValueValidator(1)])
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review by '{self.user.username}' for '{self.anime.title}'"