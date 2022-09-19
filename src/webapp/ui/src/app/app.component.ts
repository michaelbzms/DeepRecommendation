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
  movies_dict: { [id: string] : Movie; } = {}
  recommendations: Movie[] = [];
  files: any = {};

  constructor(private backendService: BackendService) {
    this.movies$.subscribe(dict => {
      this.movies = [];
      for (const [imdbID, data] of Object.entries(dict)) {
        // @ts-ignore
        let m = new Movie(imdbID, data['primaryTitle'], data['startYear'], data['genres']);
        this.movies.push(m);
        this.movies_dict[imdbID] = m;
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
    recommendations$.subscribe(recommendations => {
      console.log(recommendations);
      this.recommendations = [];
      for (let r of recommendations) {
        let m = this.movies_dict[r.imdbID];
        m.score = r.score;        // TODO: this changes score of original movies object but that should be ok?
        m.because = r.because;
        m.attention = r.attention;
        this.recommendations.push(m);
      }
    })
  }

  clear_ratings() {
    for (let movie of this.movies) {
      movie.rating = null;
    }
  }

  onRatingsUpload(event: any) {
    console.log("I was fired!");
    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target) {
        try {
          let ratings = JSON.parse((event.target.result) as string);
          // TODO
          console.log('uploaded ratings:', ratings);
        } catch (error) {
          console.error(error);
        }
      } else {
        console.error("Null target")
      }
    };
    reader.readAsText(event.files[0]);
  }

  test(event: any) {
    console.log("Error trying to upload file");
    console.error(event)
  }
}
