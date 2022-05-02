import {Component} from '@angular/core';
import {BackendService} from "./services/backend.service";
import {Movie} from "./models/movies";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'ui';
  k = 10;
  movies$ = this.backendService.getMovies();
  movies: Movie[] = []

  constructor(private backendService: BackendService) {
    this.movies$.subscribe(dict => {
      this.movies = [];
      for (const [imdbID, data] of Object.entries(dict)) {
        // @ts-ignore
        this.movies.push(new Movie(imdbID, data['primaryTitle'], data['startYear'], data['genres']));
      }
      console.log(this.movies);
    })
  }

  recommend() {
    // gather user ratings TODO: don't wait to do this now
    let user_ratings: object[] = []
    for (let movie of this.movies) {
      if (movie.rating != null) {
        user_ratings.push({
          'imdbID': movie.imdbID,
          'rating': movie.rating
        })
      }
    }
    console.log(user_ratings);

    let recommendations$ = this.backendService.getRecommendations(user_ratings, this.k)
    recommendations$.subscribe(recommendation_dict => {
      console.log(recommendation_dict);
    })
  }

}
